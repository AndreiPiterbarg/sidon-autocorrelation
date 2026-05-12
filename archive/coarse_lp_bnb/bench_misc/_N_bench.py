"""Variant N: F's tight linear + restricted-spectrum δ² bound.

Math (rigorously sound):
  |δ^T A_W δ| ≤ ‖A_W − α·11ᵀ‖_op · ‖δ‖₂²        (Σδ=0 ⇒ 11ᵀ kernel vanishes)
                ≤ op_restricted · d · h²         (‖δ‖₂² ≤ d·h² in feasible set)

  In m² units: corr_w_δ² = op_restricted · d / (4n·ell)

Variant N: corr_N = Δ_BB/(2n·ell) + min(op_restricted·d, ell_int_sum) / (4n·ell)

The min ensures variant N is always ≤ F (sound regression).

Compare survivor counts vs variant F on the M1 sweep.
"""
import os, sys, time, json
import numpy as np
import numba
from numba import njit, prange
from itertools import combinations

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F


def precompute_op_norm_restricted(d, max_ell, conv_len):
    """For each (ell, s_lo), compute op_norm of (A_W − α·11ᵀ).

    Returns array op_rest[ell, s_lo] indexed flat.  We pack as
       op_rest_flat[ell * conv_len + s_lo].
    """
    op_rest = np.zeros((max_ell + 1, conv_len), dtype=np.float64)
    n_pairs_arr = np.zeros((max_ell + 1, conv_len), dtype=np.int64)
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        for s_lo in range(n_windows):
            A = np.zeros((d, d), dtype=np.float64)
            n_pairs = 0
            for i in range(d):
                for j in range(d):
                    if s_lo <= i + j <= s_lo + ell - 2:
                        A[i, j] = 1.0
                        n_pairs += 1
            n_pairs_arr[ell, s_lo] = n_pairs
            if n_pairs == 0:
                continue
            alpha = n_pairs / (d * d)
            A_rest = A - alpha
            eigs = np.linalg.eigvalsh(A_rest)
            op_rest[ell, s_lo] = float(np.abs(eigs).max())
    return op_rest, n_pairs_arr


@njit(parallel=True, cache=True)
def prune_N(batch_int, n_half, m, c_target,
            ell_prefix, n_pairs_arr, op_rest_d_arr):
    """Variant N pruning kernel.

    Per-window correction (m² units):
       Δ_BB/(2n·ell) + min(op_rest·d, n_pairs) / (4n·ell)
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half)
    eps_margin = 1e-9 * m_d * m_d
    max_ell = 2 * d
    cs_base_m2 = c_target * m_d * m_d
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    for b in prange(B):
        # Build conv (for window-sum check)
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2*i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            if n_windows <= 0:
                continue
            scale_ell = scale_arr[ell]

            # Sliding window for ws
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]

                # Compute Δ_BB (sort+extremes) for this window
                # BB_j = Σ_{i:(i,j)∈W} c_i — fast via sliding sum on prefix
                # We compute BB per index here in O(d^2) — could prefix-sum
                # but this is the simple approach.
                BB = np.zeros(d, dtype=np.int64)
                for j in range(d):
                    for i in range(d):
                        k = i + j
                        if s_lo <= k <= s_lo + n_cv - 1:
                            BB[j] += np.int64(batch_int[b, i])
                # Sort BB descending
                BB_sorted = np.sort(BB)
                half = d // 2
                Delta_BB = np.int64(0)
                for jj in range(half):
                    Delta_BB += BB_sorted[d - 1 - jj] - BB_sorted[jj]

                # δ² bound: min(op_rest*d, n_pairs)
                n_pairs = n_pairs_arr[ell, s_lo]
                op_rest_d = op_rest_d_arr[ell, s_lo]  # = op_rest * d, precomputed
                delta_sq = min(np.float64(n_pairs), op_rest_d)

                # corr_N (m² units)
                corr_w = (np.float64(Delta_BB) / (2.0 * np.float64(n_half) * ell)
                          + delta_sq / (4.0 * np.float64(n_half) * ell))
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                dyn_it = np.int64(dyn_x)
                if ws > dyn_it:
                    pruned = True
                    break

        if pruned:
            survived[b] = False
    return survived


def run(n_half, m, c_target, batch_size=200_000, verbose=True):
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total = count_compositions(n_half, S_half)
    if verbose:
        print(f"\n=== n_half={n_half}, m={m}, c_target={c_target} ===")
        print(f"     d={d}, palindromic comps={n_total:,}")

    # Precompute window structure
    conv_len = 2 * d - 1
    max_ell = 2 * d
    print(f"     precomputing op-norm restricted for d={d}...", flush=True)
    t0 = time.time()
    op_rest, n_pairs_arr = precompute_op_norm_restricted(d, max_ell, conv_len)
    op_rest_d = op_rest * d
    # ell_prefix unused for variant N but pass dummy
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    print(f"     precompute done [{time.time()-t0:.1f}s]")

    # Warm
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S_full
    prune_N(warm, n_half, m, c_target, ell_prefix, n_pairs_arr, op_rest_d)
    prune_F(warm, n_half, m, c_target)

    # Process
    n_proc = 0
    surv_F = 0
    surv_N = 0
    n_violations = 0
    t0 = time.time()
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        sN = prune_N(batch, n_half, m, c_target,
                      ell_prefix, n_pairs_arr, op_rest_d)
        surv_F += int(sF.sum())
        surv_N += int(sN.sum())
        # Soundness: N's survivors should be subset of F's
        if not bool(np.all(sN <= sF)):
            n_violations += int(np.sum(sN & ~sF))
    elapsed = time.time() - t0
    extra = surv_F - surv_N
    pct = 100.0 * extra / max(1, surv_F)
    print(f"     F survivors: {surv_F:,}")
    print(f"     N survivors: {surv_N:,}")
    print(f"     N prunes ADDITIONAL {extra:,} ({pct:.2f}% of F survivors)")
    if n_violations > 0:
        print(f"     *** SOUNDNESS VIOLATION: {n_violations} N-survivors not in F ***")
    print(f"     wall: {elapsed:.1f}s")
    return {'n_half': n_half, 'm': m, 'c_target': c_target,
            'n_processed': n_proc, 'surv_F': surv_F, 'surv_N': surv_N,
            'n_extra_prune': extra, 'pct_extra': pct,
            'soundness_violations': n_violations,
            'wall_sec': elapsed}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    configs = [
        (3, 10, 1.20), (3, 20, 1.20),
        (3, 10, 1.28),
        (4, 10, 1.20), (4, 10, 1.28),
        (5, 5, 1.28), (5, 10, 1.28),
        (6, 5, 1.28),
    ]
    results = []
    for nh, m, c in configs:
        results.append(run(nh, m, c))
    out = os.path.join(_dir, '_N_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}\nSummary:")
    for r in results:
        print(f"  (n={r['n_half']}, m={r['m']}, c={r['c_target']}): "
              f"F={r['surv_F']:,} → N={r['surv_N']:,}  "
              f"(+{r['pct_extra']:.2f}% prune)  "
              f"violations={r['soundness_violations']}")


if __name__ == '__main__':
    main()
