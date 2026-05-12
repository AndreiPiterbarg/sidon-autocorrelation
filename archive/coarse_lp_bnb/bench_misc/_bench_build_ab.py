"""A/B: vectorized build vs naive Python-loop equivalent.

Replicates the OLD inner loops here so we can benchmark them
without reverting build.py.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import sparse as sp
from lasserre.polya_lp.build import (
    build_handelman_lp, BuildOptions, build_window_matrices,
    _coeff_W_vector,
)
from lasserre.polya_lp.poly import enum_monomials_le, index_map
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim


def naive_build(d_eff, M_mats_eff, R):
    """Old (pre-vectorization) inner-loop logic, for benchmarking only."""
    t0 = time.time()
    monos_le_R = enum_monomials_le(d_eff, R)
    n_le_R = len(monos_le_R)
    beta_to_idx = index_map(monos_le_R)
    monos_le_Rm1 = enum_monomials_le(d_eff, R - 1)
    n_q = len(monos_le_Rm1)
    n_W = len(M_mats_eff)
    n_lambda_var = n_W

    alpha_idx = 0
    lambda_start = 1
    q_start = 1 + n_W
    n_vars = q_start + n_q  # eliminate_c_slacks=True

    rows, cols, vals = [], [], []
    rhs = np.zeros(n_le_R)

    rows.append(beta_to_idx[tuple([0]*d_eff)])
    cols.append(alpha_idx); vals.append(-1.0)

    coeff_per_W = [_coeff_W_vector(M_W, beta_to_idx) for M_W in M_mats_eff]
    for w in range(n_W):
        for beta_i, coef in coeff_per_W[w].items():
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

    A = sp.csr_matrix(
        (np.asarray(vals), (np.asarray(rows, dtype=np.int64),
                            np.asarray(cols, dtype=np.int64))),
        shape=(n_le_R, n_vars),
    )
    # add simplex row (sum lambda = 1)
    sim = sp.csr_matrix(
        (np.ones(n_W), (np.zeros(n_W, dtype=np.int64),
                        np.arange(lambda_start, lambda_start + n_W, dtype=np.int64))),
        shape=(1, n_vars),
    )
    A = sp.vstack([A, sim], format="csr")
    return time.time() - t0, A.nnz


def vectorized_build(d_eff, M_mats_eff, R):
    """Use the actual build.py path."""
    opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=True)
    t0 = time.time()
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    return time.time() - t0, build.n_nonzero_A


print(f"{'d':>3} {'R':>3} {'naive_s':>10} {'vec_s':>10} {'speedup':>10} "
      f"{'naive_nnz':>10} {'vec_nnz':>10}")
print("-" * 70)
for d, R in [(16, 8), (24, 6), (24, 8), (32, 6), (32, 8)]:
    _, M_mats = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
    d_eff = z2_dim(d)
    t_n, nnz_n = naive_build(d_eff, M_mats_eff, R)
    t_v, nnz_v = vectorized_build(d_eff, M_mats_eff, R)
    sp_str = f"{t_n / t_v:.2f}x" if t_v > 0 else "inf"
    print(f"{d:>3} {R:>3} {t_n:>10.3f} {t_v:>10.3f} {sp_str:>10} "
          f"{nnz_n:>10,} {nnz_v:>10,}")
    if nnz_n != nnz_v:
        print(f"  !!! NNZ MISMATCH ({nnz_n} vs {nnz_v})")
