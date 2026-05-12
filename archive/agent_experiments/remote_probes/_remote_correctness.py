"""Correctness: confirm vectorized build produces SAME alpha as naive at scale.

For (d=24 R=8), (d=32 R=6), (d=32 R=8) we build the LP both ways and solve
both. If alphas match to MOSEK tolerance, vectorization is correctness-safe
at the scales we care about.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import sparse as sp
from lasserre.polya_lp.build import (
    build_handelman_lp, BuildOptions, build_window_matrices,
    BuildResult, _coeff_W_vector,
)
from lasserre.polya_lp.poly import enum_monomials_le, index_map
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp


def naive_full_build(d_eff, M_mats_eff, R) -> BuildResult:
    """Pre-vectorization build, returning a BuildResult so solver can run.

    eliminate_c_slacks=True (saves memory at scale).
    """
    monos_le_R = enum_monomials_le(d_eff, R)
    n_le_R = len(monos_le_R)
    beta_to_idx = index_map(monos_le_R)
    monos_le_Rm1 = enum_monomials_le(d_eff, R - 1)
    n_q = len(monos_le_Rm1)
    n_W = len(M_mats_eff)
    alpha_idx = 0
    lambda_start = 1
    q_start = 1 + n_W
    n_vars = q_start + n_q

    rows, cols, vals = [], [], []
    rows.append(beta_to_idx[tuple([0]*d_eff)])
    cols.append(alpha_idx); vals.append(-1.0)

    for w, M_W in enumerate(M_mats_eff):
        for beta_i, coef in _coeff_W_vector(M_W, beta_to_idx).items():
            rows.append(beta_i); cols.append(lambda_start + w); vals.append(coef)

    for k_idx, K in enumerate(monos_le_Rm1):
        row_K = beta_to_idx.get(K)
        if row_K is not None:
            rows.append(row_K); cols.append(q_start + k_idx); vals.append(1.0)
        for j in range(d_eff):
            K_plus = list(K); K_plus[j] += 1
            row_idx = beta_to_idx.get(tuple(K_plus))
            if row_idx is not None:
                rows.append(row_idx); cols.append(q_start + k_idx); vals.append(-1.0)

    A_ub = -sp.csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_le_R, n_vars),
    )
    b_ub = np.zeros(n_le_R)
    sim = sp.csr_matrix(
        (np.ones(n_W),
         (np.zeros(n_W, dtype=np.int64),
          np.arange(lambda_start, lambda_start + n_W, dtype=np.int64))),
        shape=(1, n_vars),
    )
    b_eq = np.array([1.0])

    bounds = [(None, None)]
    for _ in range(n_W):
        bounds.append((0.0, None))
    for _ in range(n_q):
        bounds.append((None, None))

    c_obj = np.zeros(n_vars); c_obj[alpha_idx] = -1.0

    return BuildResult(
        A_eq=sim, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, c=c_obj, bounds=bounds,
        n_vars=n_vars, alpha_idx=alpha_idx,
        lambda_idx=slice(lambda_start, lambda_start + n_W),
        q_idx=slice(q_start, q_start + n_q),
        c_idx=slice(q_start + n_q, q_start + n_q),
        monos_le_R=monos_le_R, monos_le_Rm1=monos_le_Rm1,
        n_windows=n_W, fixed_lambda=None,
        options=BuildOptions(R=R, use_z2=True, eliminate_c_slacks=True),
        build_wall_s=0.0, n_nonzero_A=A_ub.nnz,
        c_slacks_eliminated=True,
    )


print(f"{'d':>3} {'R':>3} {'naive_alpha':>14} {'vec_alpha':>14} {'diff':>10} "
      f"{'naive_t':>9} {'vec_t':>9}")
print('-' * 80)
results = []
for d, R in [(24, 8), (32, 6), (32, 8), (48, 6)]:
    _, M_mats = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
    d_eff = z2_dim(d)

    t0 = time.time()
    b_naive = naive_full_build(d_eff, M_mats_eff, R)
    sol_n = solve_lp(b_naive, solver='mosek', verbose=False)
    t_naive = time.time() - t0

    t0 = time.time()
    opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=True)
    b_vec = build_handelman_lp(d_eff, M_mats_eff, opts)
    sol_v = solve_lp(b_vec, solver='mosek', verbose=False)
    t_vec = time.time() - t0

    a_n = sol_n.alpha
    a_v = sol_v.alpha
    diff = (None if a_n is None or a_v is None else abs(a_n - a_v))
    diff_str = f"{diff:.2e}" if diff is not None else "  N/A"
    an_str = f"{a_n:.10f}" if a_n is not None else "  N/A"
    av_str = f"{a_v:.10f}" if a_v is not None else "  N/A"
    print(f"{d:>3} {R:>3} {an_str:>14} {av_str:>14} {diff_str:>10} "
          f"{t_naive:>9.2f} {t_vec:>9.2f}", flush=True)
    results.append(dict(d=d, R=R, alpha_naive=a_n, alpha_vec=a_v,
                         diff=diff, t_naive=t_naive, t_vec=t_vec))

with open('correctness_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
