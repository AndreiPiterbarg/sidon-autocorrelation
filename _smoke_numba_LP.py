"""Numba-jitted LP solver for the Q-bench min-max LP.

Target LP:
    max t   s.t.  A·λ - t·1 >= 0  ∀σ (n_sigma rows),  Σλ=1,  λ>=0,  t free.
Variables: x = [λ_0, ..., λ_{n_win-1}, t].  λ in R^{n_win}_{>=0}, t in R.

Equivalent LP (split t = t_p - t_n, t_p,t_n >= 0):
    min -t_p + t_n
    s.t.  -A·λ + (t_p - t_n) <= 0     [n_sigma rows]
           Σλ = 1                       [1 row]
          λ, t_p, t_n >= 0

Simplification.  A useful re-cast: since Σλ = 1 and λ >= 0, A·λ is a convex
combination of A's columns.  The min-max value t* = max_λ min_σ (A·λ)_σ.

We solve the DUAL (often smaller because n_win << n_sigma in practice):
    min_{μ,τ} τ
    s.t.   μ^T A  <=  τ·1_w  ∀w     (n_win rows)
            Σμ = 1, μ >= 0
By LP duality, dual_opt = primal_opt = t*.  The dual has n_sigma+1 vars
and n_win equality + n_win inequality + 1 normalization constraints.

We implement a *Big-M Bland* tableau simplex on the DUAL because:
  * variables = n_sigma + 1, but the basis size = n_win + 1, which is small.
  * we can warm-start once a basis is found.

Test plan: 50 random comps at d=6, compare with scipy.linprog HiGHS.
"""
import os, sys, time
import numpy as np
from numba import njit
from scipy.optimize import linprog

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from _Q_bench import (_enum_balanced_signs, _build_windows,
                      _composition_window_data, _q_bound_lp)


# ----------------------------------------------------------------------
# Build A matrix for one composition (matches _q_bound_lp's A construction)
# ----------------------------------------------------------------------
def build_A_matrix(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target):
    """Return A: (n_sigma, n_win) such that LP is
           max t s.t. A·λ >= t,  Σλ = 1,  λ >= 0.
    """
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    n_d = float(n_half)

    inv_4nl = np.array([1.0 / (4.0 * n_d * ell) for (ell, _) in windows])
    cs_m2 = c_target * m * m
    V = ws.astype(np.float64) * inv_4nl - ell_int_sums.astype(np.float64) * inv_4nl - cs_m2

    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]
    M_mat = sigmas.astype(np.float64) @ BB_over_ell.T   # (n_sigma, n_win)
    A = V[None, :] - M_mat / (2.0 * n_d)
    return A


# ----------------------------------------------------------------------
# Numba revised simplex (Bland's rule) for primal LP:
#     max t  s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  x_j in {>=0,free}
# We convert to standard form and use a Big-M two-phase approach.
#
# For our specific LP, we can sidestep complexity:
#   x = [λ_0, ..., λ_{n_win-1}, t_p, t_n]
#   maximize  t_p - t_n
#   subject to:
#     -A·λ + t_p - t_n <= 0      (n_sigma rows)  -> +s_i = 0  (slack >= 0)
#      Σλ = 1                                    (1 eq row)
#     all vars >= 0
#
# Standard form (min):
#   min  -t_p + t_n
#   subject to:
#     -A·λ + t_p - t_n + s_i = 0  for i = 1..n_sigma   (slacks for <= 0)
#      Σλ = 1
#     λ >= 0, t_p,t_n >= 0, s >= 0.
#
# The first n_sigma rows have RHS = 0, so slacks s_i are immediate basis.
# The Σλ = 1 row needs an artificial variable: add a_eq, basis it.
# Big-M:  min -t_p + t_n + M·a_eq
#
# Variables (in order): λ (n_win), t_p, t_n, slacks (n_sigma), artificial (1)
# Total: n_win + 2 + n_sigma + 1 = n_win + n_sigma + 3.
# Rows: n_sigma + 1.
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=False)
def _simplex_solve(A, max_iter, tol):
    """Solve  max t  s.t.  A·λ - t >= 0,  Σλ = 1,  λ >= 0,  t free.

    Encoded as min -t_p + t_n + M·art with all >= 0. Bland's rule.

    Returns (status, t_opt, n_iter):
        status=0 optimal, status=1 unbounded (shouldn't happen),
        status=2 max_iter, status=3 infeasible.
    """
    n_sigma = A.shape[0]
    n_win = A.shape[1]
    n_rows = n_sigma + 1
    # Var ordering: 0..n_win-1 = λ, n_win = t_p, n_win+1 = t_n,
    #               n_win+2 .. n_win+1+n_sigma = slacks, n_win+2+n_sigma = artificial
    n_vars = n_win + 3 + n_sigma
    art_idx = n_win + 2 + n_sigma
    tp_idx = n_win
    tn_idx = n_win + 1

    # Big M
    M = 1.0e7

    # Build augmented tableau T of shape (n_rows+1, n_vars+1):
    #   T[0:n_rows, 0:n_vars]  =  constraint matrix
    #   T[0:n_rows, -1]        =  rhs
    #   T[-1, :n_vars]         =  reduced costs (we'll fill with c - c_B B^{-1} N pattern)
    #   T[-1, -1]              =  -obj

    T = np.zeros((n_rows + 1, n_vars + 1))

    # Constraint rows 0..n_sigma-1: -A·λ + t_p - t_n + s_i = 0
    for i in range(n_sigma):
        for j in range(n_win):
            T[i, j] = -A[i, j]
        T[i, tp_idx] = 1.0
        T[i, tn_idx] = -1.0
        # slack for row i
        T[i, n_win + 2 + i] = 1.0
        T[i, -1] = 0.0

    # Equality row n_sigma: Σλ - art? No, Σλ + art = 1 (artificial >= 0)
    eq_row = n_sigma
    for j in range(n_win):
        T[eq_row, j] = 1.0
    T[eq_row, art_idx] = 1.0
    T[eq_row, -1] = 1.0

    # Initial basis: slacks for rows 0..n_sigma-1 (var indices n_win+2 .. n_win+1+n_sigma)
    # And artificial for row n_sigma (var index art_idx)
    basis = np.empty(n_rows, dtype=np.int64)
    for i in range(n_sigma):
        basis[i] = n_win + 2 + i
    basis[n_sigma] = art_idx

    # Initial objective row: c - c_B^T · A (for current basis)
    # c[j] = -1 for tp, +1 for tn, M for art, 0 else.
    # c_B for slacks = 0, c_B for art = M.
    # So obj row = c - M·(eq_row).  T[-1,j] = c[j] - M·T[eq_row,j]
    # Then T[-1,-1] = 0 - M·1 = -M (negative of cur obj contribution from art).
    # Actually for "min" tableau, we keep T[-1, j] = reduced cost = c_j - z_j.
    # z_j = c_B^T · (column j of constraint matrix in basis form). Initially basis is all-identity.
    # so z_j = c_B[i] * T[i,j] summed.  For slack basis: c_B[i]=0 for i<n_sigma, c_B[n_sigma]=M.
    # z_j = M · T[eq_row, j].
    # reduced cost = c_j - z_j = c_j - M·T[eq_row, j].

    for j in range(n_vars):
        cj = 0.0
        if j == tp_idx:
            cj = -1.0
        elif j == tn_idx:
            cj = 1.0
        elif j == art_idx:
            cj = M
        T[-1, j] = cj - M * T[eq_row, j]
    # rhs: T[-1, -1] = 0 - M·1 = -M (current value of -obj)
    T[-1, -1] = -M  # since T[-1,-1] = -(c_B^T · b)

    # Bland's rule loop
    for it in range(max_iter):
        # Find entering var: smallest index with reduced cost < -tol
        enter = -1
        for j in range(n_vars):
            if T[-1, j] < -tol:
                enter = j
                break
        if enter < 0:
            # Optimal. Check artificial in basis with positive value -> infeasible.
            for i in range(n_rows):
                if basis[i] == art_idx and T[i, -1] > tol:
                    return 3, 0.0, it
            # Extract t_opt = t_p - t_n (look up basis values)
            t_p = 0.0
            t_n = 0.0
            for i in range(n_rows):
                if basis[i] == tp_idx:
                    t_p = T[i, -1]
                elif basis[i] == tn_idx:
                    t_n = T[i, -1]
            return 0, t_p - t_n, it

        # Min ratio test: leaving row
        leave = -1
        best_ratio = 1.0e300
        # Bland: among all rows with positive coeff and min ratio, pick smallest basis index
        for i in range(n_rows):
            aij = T[i, enter]
            if aij > tol:
                r = T[i, -1] / aij
                if r < best_ratio - tol:
                    best_ratio = r
                    leave = i
                elif r < best_ratio + tol and leave >= 0:
                    # tie: pick smaller basis var index (Bland)
                    if basis[i] < basis[leave]:
                        leave = i
        if leave < 0:
            return 1, 0.0, it  # unbounded

        # Pivot on (leave, enter)
        piv = T[leave, enter]
        # normalize leave row
        for j in range(n_vars + 1):
            T[leave, j] = T[leave, j] / piv
        # eliminate from other rows including objective
        for i in range(n_rows + 1):
            if i == leave:
                continue
            factor = T[i, enter]
            if factor != 0.0:
                for j in range(n_vars + 1):
                    T[i, j] = T[i, j] - factor * T[leave, j]

        basis[leave] = enter

    return 2, 0.0, max_iter


@njit(cache=True)
def numba_lp_t_opt(A, max_iter=50000, tol=1e-9):
    status, t_opt, _ = _simplex_solve(A, max_iter, tol)
    return status, t_opt


# ----------------------------------------------------------------------
# Test driver
# ----------------------------------------------------------------------
def gen_random_comp(d, sum_target, rng):
    """Generate random nonneg integer composition c of length d summing to sum_target."""
    cuts = rng.integers(0, sum_target + 1, size=d - 1)
    cuts.sort()
    parts = np.empty(d, dtype=np.int32)
    parts[0] = cuts[0]
    for i in range(1, d - 1):
        parts[i] = cuts[i] - cuts[i - 1]
    parts[-1] = sum_target - cuts[-1]
    return parts


def main():
    n_half, m, c_target = 3, 5, 1.20
    d = 2 * n_half
    S_full = 4 * n_half * m
    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    n_win = len(windows)
    n_sigma = len(sigmas)
    print(f"d={d}, n_win={n_win}, n_sigma={n_sigma}")

    # Warm up jit
    rng = np.random.default_rng(0)
    c0 = gen_random_comp(d, S_full, rng)
    A0 = build_A_matrix(c0, windows, ell_int_sums, sigmas, n_half, m, c_target)
    print("Warming up numba...")
    t_warm = time.time()
    status, t_opt = numba_lp_t_opt(A0)
    print(f"  warm: status={status}, t_opt={t_opt:.6f}, compile time {time.time()-t_warm:.2f}s")

    # Correctness: 50 random comps at d=6
    print("\n=== Correctness (50 random comps at d=6) ===")
    n_test = 50
    diffs = []
    rng = np.random.default_rng(42)
    sound_violations = 0
    worst_under = 0.0  # numba below scipy by how much
    for k in range(n_test):
        c_int = gen_random_comp(d, S_full, rng)
        A = build_A_matrix(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target)

        # Numba
        status, t_n = numba_lp_t_opt(A)
        if status != 0:
            print(f"  k={k}: numba status={status} (skip)")
            continue

        # scipy
        t_s, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                             n_half, m, c_target)
        if not np.isfinite(t_s):
            continue

        diff = t_n - t_s
        diffs.append(diff)
        # Soundness: numba >= scipy (slightly under is ok if pruning rule allows; but
        # under means MISSED prunes - we accept tolerance up to 1e-6 m^2)
        if diff < -1e-6:
            sound_violations += 1
            if abs(diff) > worst_under:
                worst_under = abs(diff)

    diffs = np.array(diffs)
    print(f"  ran {len(diffs)} comps")
    print(f"  max |numba - scipy|     = {np.max(np.abs(diffs)):.3e}")
    print(f"  mean (numba - scipy)    = {np.mean(diffs):+.3e}")
    print(f"  numba < scipy violations (>1e-6): {sound_violations}")
    print(f"  worst under-approximation        = {worst_under:.3e}")

    # Speed test: 100 LPs at d=6,8,10,12
    speedup_dict = {}
    for d_test in [6, 8, 10, 12]:
        n_h = d_test // 2
        m_t = 5
        windows_t, ell_int_sums_t = _build_windows(d_test)
        sigmas_t = _enum_balanced_signs(d_test)
        n_win_t = len(windows_t)
        n_sigma_t = len(sigmas_t)

        n_lp = 100 if d_test <= 10 else 30
        rng = np.random.default_rng(7)
        c_list = [gen_random_comp(d_test, 4 * n_h * m_t, rng) for _ in range(n_lp)]
        A_list = [build_A_matrix(c, windows_t, ell_int_sums_t, sigmas_t,
                                 n_h, m_t, c_target) for c in c_list]

        # Warm up
        _ = numba_lp_t_opt(A_list[0])

        # numba
        t0 = time.time()
        for A in A_list:
            _ = numba_lp_t_opt(A)
        t_numba = time.time() - t0

        # scipy
        t0 = time.time()
        for c in c_list:
            _ = _q_bound_lp(c, windows_t, ell_int_sums_t, sigmas_t,
                            n_h, m_t, c_target)
        t_scipy = time.time() - t0

        print(f"\n=== Speed at d={d_test} (n_win={n_win_t}, n_sigma={n_sigma_t}) ===")
        print(f"  numba:  {t_numba*1000/n_lp:7.2f} ms/LP   (total {t_numba:.2f}s)")
        print(f"  scipy:  {t_scipy*1000/n_lp:7.2f} ms/LP   (total {t_scipy:.2f}s)")
        speedup = t_scipy / max(t_numba, 1e-9)
        print(f"  speedup: {speedup:.2f}x")
        speedup_dict[d_test] = (speedup, t_numba * 1000 / n_lp, t_scipy * 1000 / n_lp)

    print(f"\nFINAL:")
    for d_test, (sp, tn_ms, ts_ms) in speedup_dict.items():
        print(f"  d={d_test}: speedup={sp:.2f}x, numba={tn_ms:.2f} ms/LP, scipy={ts_ms:.2f} ms/LP")
    print(f"  worst under-approx (m^2): {worst_under:.3e}")
    print(f"  prune threshold (m^2):    {1e-9 * m_t * m_t:.3e}")
    return speedup_dict, worst_under, sound_violations


if __name__ == '__main__':
    sp_dict, worst, viol = main()
