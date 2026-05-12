"""Scaling benchmark for the vectorized LP build.

Measures: build wall-time, n_vars, n_rows, nnz, peak RSS.
A/B compares vectorized build vs the naive (pre-vectorization) loop.
Runs each (d, R) until budget exhausted.
"""
import sys, os, time, gc, tracemalloc, json, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import sparse as sp
from lasserre.polya_lp.build import (
    build_handelman_lp, BuildOptions, build_window_matrices,
    _coeff_W_vector,
)
from lasserre.polya_lp.poly import enum_monomials_le, index_map
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def naive_build(d_eff, M_mats_eff, R, eliminate=True):
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
    sim = sp.csr_matrix(
        (np.ones(n_W), (np.zeros(n_W, dtype=np.int64),
                        np.arange(lambda_start, lambda_start + n_W, dtype=np.int64))),
        shape=(1, n_vars),
    )
    A = sp.vstack([A, sim], format="csr")
    return time.time() - t0, A.nnz, n_le_R, n_vars


def vectorized_build(d_eff, M_mats_eff, R):
    opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=True)
    t0 = time.time()
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    t = time.time() - t0
    nrows = (build.A_eq.shape[0] if build.A_eq is not None else 0) + \
            (build.A_ub.shape[0] if build.A_ub is not None else 0)
    return t, build.n_nonzero_A, nrows, build.n_vars


SCHEDULE = [
    (16, 8), (16, 10), (16, 12),
    (24, 6), (24, 8), (24, 10),
    (32, 6), (32, 8), (32, 10),
    (48, 4), (48, 6), (48, 8),
    (64, 4), (64, 6), (64, 8),
    (80, 4), (80, 6),
    (96, 4),
    (128, 4),
]

PER_BUILD_TIMEOUT = 600  # 10 min per build

results = []
for d, R in SCHEDULE:
    print(f"\n=== d={d}, R={R} ===", flush=True)
    rss0 = rss_mb()
    try:
        _, M_mats = build_window_matrices(d)
        M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
        d_eff = z2_dim(d)
    except Exception as e:
        print(f"  setup fail: {e}", flush=True)
        results.append(dict(d=d, R=R, err=str(e)))
        continue

    rec = dict(d=d, R=R, d_eff=d_eff, n_W=len(M_mats_eff))

    # Vectorized first.
    gc.collect()
    rss_pre = rss_mb()
    try:
        t_v, nnz_v, nrows_v, nvars_v = vectorized_build(d_eff, M_mats_eff, R)
        rec.update(vec_s=t_v, nnz=nnz_v, nrows=nrows_v, nvars=nvars_v,
                   vec_rss_mb=rss_mb() - rss_pre)
        print(f"  vec: {t_v:.2f}s  nrows={nrows_v:,}  nvars={nvars_v:,}  "
              f"nnz={nnz_v:,}  drss={rec['vec_rss_mb']:.0f}MB", flush=True)
    except MemoryError:
        rec["vec_err"] = "OOM"
        print("  vec OOM", flush=True)
        results.append(rec)
        continue
    except Exception as e:
        rec["vec_err"] = str(e)
        print(f"  vec err: {e}", flush=True)
        results.append(rec)
        continue

    # Naive for comparison (skip if vec already > 60s; naive is at least as slow).
    gc.collect()
    if t_v < 90.0 and nrows_v < 5_000_000:  # skip giant cases for naive
        rss_pre = rss_mb()
        try:
            t_n, nnz_n, nrows_n, nvars_n = naive_build(d_eff, M_mats_eff, R)
            rec.update(naive_s=t_n,
                       speedup=(t_n / t_v if t_v > 0 else None),
                       nnz_n=nnz_n)
            print(f"  naive: {t_n:.2f}s  speedup={rec['speedup']:.2f}x", flush=True)
        except Exception as e:
            rec["naive_err"] = str(e)
            print(f"  naive err: {e}", flush=True)
    else:
        rec["naive_skipped"] = "too_large"

    results.append(rec)
    # checkpoint
    with open("scale_bench_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

print("\n\n=== SUMMARY ===", flush=True)
print(f"{'d':>3} {'R':>3} {'d_eff':>5} {'n_W':>5} {'nrows':>10} "
      f"{'nvars':>11} {'nnz':>11} {'vec_s':>8} {'naive_s':>8} {'speedup':>8}")
print("-" * 100)
for r in results:
    if "vec_err" in r:
        print(f"{r['d']:>3} {r['R']:>3} {r.get('d_eff','?'):>5}    *  "
              f"vec_err: {r['vec_err']}")
        continue
    sp_str = f"{r.get('speedup', 0):.2f}x" if r.get('speedup') else "  -- "
    naive_str = f"{r.get('naive_s', 0):.2f}" if 'naive_s' in r else "  -- "
    print(f"{r['d']:>3} {r['R']:>3} {r['d_eff']:>5} {r['n_W']:>5} "
          f"{r['nrows']:>10,} {r['nvars']:>11,} {r['nnz']:>11,} "
          f"{r['vec_s']:>8.2f} {naive_str:>8} {sp_str:>8}")

with open("scale_bench_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nWrote scale_bench_results.json")
