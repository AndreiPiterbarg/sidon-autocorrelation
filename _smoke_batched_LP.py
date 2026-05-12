"""Q-LP speedup smoke test: scipy.linprog vs highspy direct.

Goal: bypass scipy's Python overhead and call HiGHS via highspy directly.
Pre-build the static parts of the LP once per d, only update cost / V_w
terms per composition.

Test set: F-survivors at d=8 (n_half=4, m=10, c_target=1.28) — ~1014 LPs.
Soundness: t_opt values must match within 1e-9.
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'cloninger-steinerberger', 'cpu'))

import scipy
from scipy.optimize import linprog
import highspy

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import (_build_windows, _enum_balanced_signs,
                       _composition_window_data, _q_bound_lp)


# ----------------------------------------------------------------------
# Reference: scipy.linprog (current implementation)
# ----------------------------------------------------------------------
def q_bound_scipy(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target):
    """Mirror of _q_bound_lp using scipy.linprog."""
    return _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                       n_half, m, c_target)


# ----------------------------------------------------------------------
# highspy direct path
# ----------------------------------------------------------------------
class HighsQSolver:
    """Per-d_child solver: build the σ matrix once, reuse columns / sparsity."""

    def __init__(self, d, windows, ell_int_sums, sigmas, n_half, m, c_target):
        self.d = d
        self.windows = windows
        self.ell_int_sums = ell_int_sums
        self.sigmas = sigmas
        self.n_half = n_half
        self.m = m
        self.c_target = c_target
        self.n_win = len(windows)
        self.n_sigma = len(sigmas)
        self.nvar = self.n_win + 1   # λ_0..λ_{n_win-1}, t

        # Pre-compute static arrays:
        self.n_d = float(n_half)
        self.inv_4nl = np.array([1.0 / (4.0 * self.n_d * ell)
                                   for (ell, _) in windows], dtype=np.float64)
        self.inv_2nl = np.array([1.0 / (2.0 * self.n_d * ell)
                                   for (ell, _) in windows], dtype=np.float64)
        self.cs_m2 = c_target * m * m
        self.ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)

        # Pre-build LP scaffolding (column / row bounds, objective, sparsity).
        # Constraint layout (rowwise):
        #   rows 0 .. n_sigma-1: -A[σ,:] · λ + t  <=  0   (i.e. row_lower=-inf, row_upper=0)
        #     equivalently:    A[σ,:]·λ - t  >= 0
        #   row n_sigma:         sum λ = 1
        # We use row_lower / row_upper bounds.

        # Column bounds: λ_w >= 0 (upper +inf); t free (-inf, +inf)
        self.col_lower = np.empty(self.nvar, dtype=np.float64)
        self.col_upper = np.empty(self.nvar, dtype=np.float64)
        self.col_lower[:self.n_win] = 0.0
        self.col_upper[:self.n_win] = highspy.kHighsInf
        self.col_lower[self.n_win] = -highspy.kHighsInf
        self.col_upper[self.n_win] = highspy.kHighsInf

        # Objective: min -t
        self.col_cost = np.zeros(self.nvar, dtype=np.float64)
        self.col_cost[self.n_win] = -1.0

        # Row bounds:
        self.row_lower = np.empty(self.n_sigma + 1, dtype=np.float64)
        self.row_upper = np.empty(self.n_sigma + 1, dtype=np.float64)
        # rows 0..n_sigma-1: -A[σ,:]·λ + t  <=  0
        self.row_lower[:self.n_sigma] = -highspy.kHighsInf
        self.row_upper[:self.n_sigma] = 0.0
        # row n_sigma: sum λ == 1
        self.row_lower[self.n_sigma] = 1.0
        self.row_upper[self.n_sigma] = 1.0

        # Pre-build sparsity (rowwise): n_sigma rows of (n_win+1) nonzeros each,
        # plus 1 row of n_win nonzeros.  All rows are dense in λ + t.
        # row r has indices [0..n_win-1, n_win] for r < n_sigma,
        # row n_sigma has indices [0..n_win-1].
        nnz = self.n_sigma * (self.n_win + 1) + self.n_win
        self.A_start = np.empty(self.n_sigma + 2, dtype=np.int32)
        self.A_index = np.empty(nnz, dtype=np.int32)
        # values updated per LP
        self.A_value_template = np.empty(nnz, dtype=np.float64)

        for r in range(self.n_sigma):
            self.A_start[r] = r * (self.n_win + 1)
            base = r * (self.n_win + 1)
            self.A_index[base:base + self.n_win] = np.arange(self.n_win, dtype=np.int32)
            self.A_index[base + self.n_win] = self.n_win   # t column
            # value for t column: +1 (the "+t" in the constraint)
            self.A_value_template[base + self.n_win] = 1.0
        # last row (sum=1)
        self.A_start[self.n_sigma] = self.n_sigma * (self.n_win + 1)
        self.A_start[self.n_sigma + 1] = nnz
        last_base = self.n_sigma * (self.n_win + 1)
        self.A_index[last_base:last_base + self.n_win] = np.arange(self.n_win, dtype=np.int32)
        self.A_value_template[last_base:last_base + self.n_win] = 1.0

        # Pre-allocate LP object (reused across solves)
        self.lp = highspy.HighsLp()
        self.lp.num_col_ = self.nvar
        self.lp.num_row_ = self.n_sigma + 1
        self.lp.col_cost_ = self.col_cost
        self.lp.col_lower_ = self.col_lower
        self.lp.col_upper_ = self.col_upper
        self.lp.row_lower_ = self.row_lower
        self.lp.row_upper_ = self.row_upper
        self.lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        self.lp.a_matrix_.start_ = self.A_start
        self.lp.a_matrix_.index_ = self.A_index
        # value_ assigned per-solve

    def solve(self, c_int):
        """Solve Q-LP for one composition; return (t_opt, lambda_opt)."""
        ws, BB = _composition_window_data(c_int, self.windows,
                                            self.n_half, self.m)
        # Build A[σ, w] = V_w - M[σ,w]/(2n)
        V = ws.astype(np.float64) * self.inv_4nl - \
            self.ell_int_sums.astype(np.float64) * self.inv_4nl - self.cs_m2
        BB_over_ell = BB.astype(np.float64) / self.ell_arr[:, None]   # (n_win, d)
        M = self.sigmas.astype(np.float64) @ BB_over_ell.T            # (n_sigma, n_win)
        A = V[None, :] - M / (2.0 * self.n_d)                         # (n_sigma, n_win)

        # Constraint:  -A[σ, :]·λ + t <= 0
        # So row r value entries [0..n_win-1] = -A[r, :]; entry n_win = +1 (template).
        # Update A_value array
        A_value = self.A_value_template.copy()
        for r in range(self.n_sigma):
            base = r * (self.n_win + 1)
            A_value[base:base + self.n_win] = -A[r, :]

        self.lp.a_matrix_.value_ = A_value

        h = highspy.Highs()
        h.silent()
        h.passModel(self.lp)
        st = h.run()
        if st != highspy.HighsStatus.kOk:
            return -np.inf, None
        ms = h.getModelStatus()
        if ms != highspy.HighsModelStatus.kOptimal:
            return -np.inf, None
        sol = h.getSolution()
        x = np.asarray(sol.col_value)
        t_opt = -h.getObjectiveValue()
        return float(t_opt), x[:self.n_win]


class HighsQSolverReused:
    """Same as above but tries to reuse a single Highs() instance via clearModel
    + passModel.  This is the "fast path"."""

    def __init__(self, d, windows, ell_int_sums, sigmas, n_half, m, c_target):
        self.d = d
        self.windows = windows
        self.ell_int_sums = ell_int_sums
        self.sigmas = sigmas
        self.n_half = n_half
        self.m = m
        self.c_target = c_target
        self.n_win = len(windows)
        self.n_sigma = len(sigmas)
        self.nvar = self.n_win + 1

        self.n_d = float(n_half)
        self.inv_4nl = np.array([1.0 / (4.0 * self.n_d * ell)
                                   for (ell, _) in windows], dtype=np.float64)
        self.cs_m2 = c_target * m * m
        self.ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)

        self.col_lower = np.empty(self.nvar, dtype=np.float64)
        self.col_upper = np.empty(self.nvar, dtype=np.float64)
        self.col_lower[:self.n_win] = 0.0
        self.col_upper[:self.n_win] = highspy.kHighsInf
        self.col_lower[self.n_win] = -highspy.kHighsInf
        self.col_upper[self.n_win] = highspy.kHighsInf

        self.col_cost = np.zeros(self.nvar, dtype=np.float64)
        self.col_cost[self.n_win] = -1.0

        self.row_lower = np.empty(self.n_sigma + 1, dtype=np.float64)
        self.row_upper = np.empty(self.n_sigma + 1, dtype=np.float64)
        self.row_lower[:self.n_sigma] = -highspy.kHighsInf
        self.row_upper[:self.n_sigma] = 0.0
        self.row_lower[self.n_sigma] = 1.0
        self.row_upper[self.n_sigma] = 1.0

        nnz = self.n_sigma * (self.n_win + 1) + self.n_win
        self.A_start = np.empty(self.n_sigma + 2, dtype=np.int32)
        self.A_index = np.empty(nnz, dtype=np.int32)
        self.A_value_template = np.empty(nnz, dtype=np.float64)

        for r in range(self.n_sigma):
            self.A_start[r] = r * (self.n_win + 1)
            base = r * (self.n_win + 1)
            self.A_index[base:base + self.n_win] = np.arange(self.n_win, dtype=np.int32)
            self.A_index[base + self.n_win] = self.n_win
            self.A_value_template[base + self.n_win] = 1.0
        self.A_start[self.n_sigma] = self.n_sigma * (self.n_win + 1)
        self.A_start[self.n_sigma + 1] = nnz
        last_base = self.n_sigma * (self.n_win + 1)
        self.A_index[last_base:last_base + self.n_win] = np.arange(self.n_win, dtype=np.int32)
        self.A_value_template[last_base:last_base + self.n_win] = 1.0

        # Pre-allocate persistent Highs() instance
        self.h = highspy.Highs()
        self.h.silent()
        # set log to off
        self.h.setOptionValue('output_flag', False)
        self.h.setOptionValue('presolve', 'on')

    def solve(self, c_int):
        ws, BB = _composition_window_data(c_int, self.windows,
                                            self.n_half, self.m)
        V = ws.astype(np.float64) * self.inv_4nl - \
            self.ell_int_sums.astype(np.float64) * self.inv_4nl - self.cs_m2
        BB_over_ell = BB.astype(np.float64) / self.ell_arr[:, None]
        M = self.sigmas.astype(np.float64) @ BB_over_ell.T
        A = V[None, :] - M / (2.0 * self.n_d)

        A_value = self.A_value_template.copy()
        for r in range(self.n_sigma):
            base = r * (self.n_win + 1)
            A_value[base:base + self.n_win] = -A[r, :]

        # Build LP
        lp = highspy.HighsLp()
        lp.num_col_ = self.nvar
        lp.num_row_ = self.n_sigma + 1
        lp.col_cost_ = self.col_cost
        lp.col_lower_ = self.col_lower
        lp.col_upper_ = self.col_upper
        lp.row_lower_ = self.row_lower
        lp.row_upper_ = self.row_upper
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = self.A_start
        lp.a_matrix_.index_ = self.A_index
        lp.a_matrix_.value_ = A_value

        self.h.clearModel()
        self.h.passModel(lp)
        st = self.h.run()
        if st != highspy.HighsStatus.kOk:
            return -np.inf, None
        ms = self.h.getModelStatus()
        if ms != highspy.HighsModelStatus.kOptimal:
            return -np.inf, None
        sol = self.h.getSolution()
        x = np.asarray(sol.col_value)
        t_opt = -self.h.getObjectiveValue()
        return float(t_opt), x[:self.n_win]


# ----------------------------------------------------------------------
# Profile breakdown of scipy.linprog: numpy build vs HiGHS time
# ----------------------------------------------------------------------
def profile_scipy_split(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target):
    """Measure (numpy build, HiGHS via scipy) split for a single composition."""
    d = len(c_int)
    n_win = len(windows)
    n_sigma = len(sigmas)
    n_d = float(n_half)
    inv_4nl = np.array([1.0 / (4.0 * n_d * ell) for (ell, _) in windows])
    cs_m2 = c_target * m * m
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)

    t_build_start = time.perf_counter()
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    V = ws.astype(np.float64) * inv_4nl - \
        ell_int_sums.astype(np.float64) * inv_4nl - cs_m2
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]
    M = sigmas.astype(np.float64) @ BB_over_ell.T
    A = V[None, :] - M / (2.0 * n_d)

    nvar = n_win + 1
    c_obj = np.zeros(nvar)
    c_obj[-1] = -1.0
    A_ub = np.zeros((n_sigma, nvar))
    A_ub[:, :n_win] = -A
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_sigma)
    A_eq = np.zeros((1, nvar))
    A_eq[0, :n_win] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n_win + [(None, None)]
    t_build = time.perf_counter() - t_build_start

    t_solve_start = time.perf_counter()
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs', options={'presolve': True})
    t_solve = time.perf_counter() - t_solve_start

    return t_build, t_solve, -res.fun if res.success else -np.inf


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    n_half, m, c_target = 4, 10, 1.28
    d = 2 * n_half
    S_half = 2 * n_half * m

    print(f"\n=== Smoke test: scipy.linprog vs highspy direct ===")
    print(f"  d={d}, n_half={n_half}, m={m}, c_target={c_target}")

    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    print(f"  n_win={len(windows)} n_sigma={len(sigmas)}")

    # Warm up F kernel
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    # Generate F-survivors
    f_survivors = []
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        sF = prune_F(batch, n_half, m, c_target)
        for idx in np.where(sF)[0]:
            f_survivors.append(batch[idx].copy())
    n_lps = len(f_survivors)
    print(f"  F-survivors: {n_lps}")

    # ---------------------------------------------------------------
    # Profile scipy split: numpy build vs HiGHS
    # ---------------------------------------------------------------
    print("\n--- scipy.linprog: build vs solve split (sample 50) ---")
    t_b_total = 0.0
    t_s_total = 0.0
    sample = f_survivors[:50]
    for c_int in sample:
        tb, ts, _ = profile_scipy_split(c_int, windows, ell_int_sums, sigmas,
                                          n_half, m, c_target)
        t_b_total += tb
        t_s_total += ts
    n_s = len(sample)
    print(f"  per-LP build: {1000*t_b_total/n_s:.3f} ms  "
          f"({100*t_b_total/(t_b_total+t_s_total):.1f}%)")
    print(f"  per-LP solve: {1000*t_s_total/n_s:.3f} ms  "
          f"({100*t_s_total/(t_b_total+t_s_total):.1f}%)")

    # ---------------------------------------------------------------
    # Run scipy.linprog over all F-survivors
    # ---------------------------------------------------------------
    print("\n--- scipy.linprog full run ---")
    t_scipy_results = np.empty(n_lps, dtype=np.float64)
    t0 = time.perf_counter()
    for k, c_int in enumerate(f_survivors):
        t_opt, _ = q_bound_scipy(c_int, windows, ell_int_sums, sigmas,
                                   n_half, m, c_target)
        t_scipy_results[k] = t_opt
    t_scipy = time.perf_counter() - t0
    print(f"  total: {t_scipy:.3f}s   per-LP: {1000*t_scipy/n_lps:.3f} ms")

    # ---------------------------------------------------------------
    # Run highspy direct (fresh Highs() per LP)
    # ---------------------------------------------------------------
    print("\n--- highspy direct (fresh Highs per LP) ---")
    solver = HighsQSolver(d, windows, ell_int_sums, sigmas,
                            n_half, m, c_target)
    t_hs_results = np.empty(n_lps, dtype=np.float64)
    t0 = time.perf_counter()
    for k, c_int in enumerate(f_survivors):
        t_opt, _ = solver.solve(c_int)
        t_hs_results[k] = t_opt
    t_hs = time.perf_counter() - t0
    print(f"  total: {t_hs:.3f}s   per-LP: {1000*t_hs/n_lps:.3f} ms")

    # ---------------------------------------------------------------
    # Run highspy direct (reused Highs instance)
    # ---------------------------------------------------------------
    print("\n--- highspy direct (reused Highs instance) ---")
    solver_r = HighsQSolverReused(d, windows, ell_int_sums, sigmas,
                                    n_half, m, c_target)
    t_hsr_results = np.empty(n_lps, dtype=np.float64)
    t0 = time.perf_counter()
    for k, c_int in enumerate(f_survivors):
        t_opt, _ = solver_r.solve(c_int)
        t_hsr_results[k] = t_opt
    t_hsr = time.perf_counter() - t0
    print(f"  total: {t_hsr:.3f}s   per-LP: {1000*t_hsr/n_lps:.3f} ms")

    # ---------------------------------------------------------------
    # Soundness: t_opt match
    # ---------------------------------------------------------------
    diff_hs = np.max(np.abs(t_scipy_results - t_hs_results))
    diff_hsr = np.max(np.abs(t_scipy_results - t_hsr_results))
    print(f"\n--- Soundness ---")
    print(f"  max |scipy - highspy(fresh)|:   {diff_hs:.3e}")
    print(f"  max |scipy - highspy(reused)|:  {diff_hsr:.3e}")
    sound = diff_hs < 1e-9 and diff_hsr < 1e-9

    speedup_fresh = t_scipy / t_hs
    speedup_reused = t_scipy / t_hsr
    print(f"\n  speedup (fresh):  {speedup_fresh:.2f}x")
    print(f"  speedup (reused): {speedup_reused:.2f}x")

    out = {
        'config': {'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
                    'n_win': len(windows), 'n_sigma': len(sigmas),
                    'n_lps': n_lps},
        'profile_scipy': {
            'per_lp_build_ms': 1000 * t_b_total / n_s,
            'per_lp_solve_ms': 1000 * t_s_total / n_s,
            'build_pct': 100 * t_b_total / (t_b_total + t_s_total),
        },
        'scipy_total_s': t_scipy,
        'scipy_per_lp_ms': 1000 * t_scipy / n_lps,
        'highspy_fresh_total_s': t_hs,
        'highspy_fresh_per_lp_ms': 1000 * t_hs / n_lps,
        'highspy_reused_total_s': t_hsr,
        'highspy_reused_per_lp_ms': 1000 * t_hsr / n_lps,
        'speedup_fresh': speedup_fresh,
        'speedup_reused': speedup_reused,
        'soundness_max_diff_fresh': diff_hs,
        'soundness_max_diff_reused': diff_hsr,
        'sound': bool(sound),
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '_smoke_batched_LP.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")
    return out


if __name__ == '__main__':
    main()
