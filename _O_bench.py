"""Variant O: F's tight linear (one-sided LP) + N's spectral δ² bound.

DERIVATION:
==========
Cascade hypotheses on ε_j := c_j/m - a_j:
  (1) |ε_j| ≤ 1/m     (C&S Lemma 2)
  (2) Σ ε = 0          (cascade total-mass invariant: ∫f = 1)
  (3) ε_j ≤ 0 if c_j = 0   (from a_j ≥ 0 — the NEW constraint)

Variant F's linear bound: max Σ ε_j BB_j subject to (1) + (2):
   Δ_BB = (1/m) · [Σ_top(d/2) BB - Σ_bot(d/2) BB]    (sort+extremes)

Variant O's linear bound: max Σ ε_j BB_j subject to (1) + (2) + (3):
   Δ_OS = (1/m) · max_{k=1..min(|T|, d//2)}
              [Σ_{top-k of T(c)} BB - Σ_{bot-k of [d]\Pos} BB]
where T(c) := {j : c_j > 0}.

Δ_OS ≤ Δ_BB always (smaller polytope), strict when c is sparse.

In m² units, corr_O linear = Δ_OS_int / (2 n_half · ell), where Δ_OS_int is
the integer-LP value (BB measured in c-units, ε scaled out by 1/m).

δ² bound: same as N (spectral, with min vs ell_int_sum fallback).

  corr_O = Δ_OS/(2n·ell) + min(op_rest·d, ell_int_sum) / (4n·ell)
"""
import os, sys, time, json
import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _N_bench import prune_N, precompute_op_norm_restricted


@njit(parallel=True, cache=True)
def prune_O(batch_int, n_half, m, c_target, n_pairs_arr, op_rest_d_arr):
    """Variant O pruning kernel.

    Per-window correction (m² units):
       Δ_OS/(2n·ell) + min(op_rest·d, n_pairs) / (4n·ell)
    where Δ_OS is the one-sided LP value (max Σ ε_j BB_j over polytope
    {|ε_j| ≤ 1, ε_j ≤ 0 if c_j = 0, Σε = 0}, in integer ε-units).
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
        # Build conv (for ws threshold check)
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

            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]

                # ===== Compute BB[j] = Σ_{i: i+j ∈ window} c[i] =====
                BB = np.zeros(d, dtype=np.int64)
                for j in range(d):
                    for i in range(d):
                        k = i + j
                        if s_lo <= k <= s_lo + n_cv - 1:
                            BB[j] += np.int64(batch_int[b, i])

                # ===== Compute Δ_OS (one-sided LP) =====
                # Sort all indices by BB ascending
                BB_sorted_asc_idx = np.argsort(BB)

                # Build T_sorted_desc_idx: T-members in BB-descending order
                T_count = 0
                for j in range(d):
                    if batch_int[b, j] > 0:
                        T_count += 1
                T_sorted_desc_idx = np.zeros(T_count, dtype=np.int64)
                ti = 0
                # Walk BB_sorted_asc_idx in reverse (BB descending), pick T members
                for jj in range(d - 1, -1, -1):
                    idx = BB_sorted_asc_idx[jj]
                    if batch_int[b, idx] > 0:
                        T_sorted_desc_idx[ti] = idx
                        ti += 1
                        if ti == T_count:
                            break

                # Iterate k = 1, 2, ..., min(T_count, d//2)
                k_max = min(T_count, d // 2)
                Delta_OS = np.int64(0)
                pos_sum = np.int64(0)
                used_pos = np.zeros(d, dtype=numba.boolean)

                for k in range(1, k_max + 1):
                    new_pos = T_sorted_desc_idx[k - 1]
                    pos_sum += BB[new_pos]
                    used_pos[new_pos] = True
                    # Accumulate bot-k of [d]\used_pos
                    neg_sum = np.int64(0)
                    found = 0
                    for jj in range(d):
                        idx = BB_sorted_asc_idx[jj]
                        if not used_pos[idx]:
                            neg_sum += BB[idx]
                            found += 1
                            if found == k:
                                break
                    if found < k:
                        break  # not enough indices remaining
                    val = pos_sum - neg_sum
                    if val > Delta_OS:
                        Delta_OS = val

                # ===== δ² bound (same as N: min of spectral and elementwise) =====
                n_pairs = n_pairs_arr[ell, s_lo]
                op_rest_d = op_rest_d_arr[ell, s_lo]
                delta_sq = min(np.float64(n_pairs), op_rest_d)

                # corr_O (m² units)
                corr_w = (np.float64(Delta_OS) / (2.0 * np.float64(n_half) * ell)
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

    conv_len = 2 * d - 1
    max_ell = 2 * d
    print(f"     precomputing op-norm restricted for d={d}...", flush=True)
    t0 = time.time()
    op_rest, n_pairs_arr = precompute_op_norm_restricted(d, max_ell, conv_len)
    op_rest_d = op_rest * d
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)  # dummy for prune_N call
    print(f"     precompute done [{time.time()-t0:.1f}s]")

    # Warm
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S_full
    prune_F(warm, n_half, m, c_target)
    prune_N(warm, n_half, m, c_target, ell_prefix, n_pairs_arr, op_rest_d)
    prune_O(warm, n_half, m, c_target, n_pairs_arr, op_rest_d)

    # Process
    n_proc = 0
    surv_F = 0
    surv_N = 0
    surv_O = 0
    n_violations_NF = 0   # N should be ⊆ F
    n_violations_ON = 0   # O should be ⊆ N
    n_violations_OF = 0   # O should be ⊆ F
    t0 = time.time()
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        sN = prune_N(batch, n_half, m, c_target, ell_prefix, n_pairs_arr, op_rest_d)
        sO = prune_O(batch, n_half, m, c_target, n_pairs_arr, op_rest_d)
        surv_F += int(sF.sum())
        surv_N += int(sN.sum())
        surv_O += int(sO.sum())
        # Soundness: each pruning method should be ⊆ the looser one
        n_violations_NF += int(np.sum(sN & ~sF))
        n_violations_ON += int(np.sum(sO & ~sN))
        n_violations_OF += int(np.sum(sO & ~sF))
    elapsed = time.time() - t0
    extra_NF = surv_F - surv_N
    extra_ON = surv_N - surv_O
    extra_OF = surv_F - surv_O
    pct_NF = 100.0 * extra_NF / max(1, surv_F)
    pct_ON = 100.0 * extra_ON / max(1, surv_N)
    pct_OF = 100.0 * extra_OF / max(1, surv_F)
    print(f"     F survivors: {surv_F:,}")
    print(f"     N survivors: {surv_N:,}  (N prunes +{extra_NF:,} = {pct_NF:.2f}% of F)")
    print(f"     O survivors: {surv_O:,}  (O prunes +{extra_ON:,} = {pct_ON:.2f}% of N, +{pct_OF:.2f}% of F)")
    if n_violations_NF + n_violations_ON + n_violations_OF > 0:
        print(f"     *** SOUNDNESS VIOLATIONS: NF={n_violations_NF}, ON={n_violations_ON}, OF={n_violations_OF} ***")
    print(f"     wall: {elapsed:.1f}s")
    return {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'n_processed': n_proc,
        'surv_F': surv_F, 'surv_N': surv_N, 'surv_O': surv_O,
        'extra_NF': extra_NF, 'extra_ON': extra_ON, 'extra_OF': extra_OF,
        'pct_NF': pct_NF, 'pct_ON': pct_ON, 'pct_OF': pct_OF,
        'soundness_violations_NF': n_violations_NF,
        'soundness_violations_ON': n_violations_ON,
        'soundness_violations_OF': n_violations_OF,
        'wall_sec': elapsed,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='small subset for fast bench')
    args = ap.parse_args()

    if args.quick:
        configs = [
            (3, 10, 1.28),
            (4, 10, 1.28),
            (5, 5, 1.28),
        ]
    else:
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
    out = os.path.join(_dir, '_O_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}\nSummary:")
    for r in results:
        print(f"  (n={r['n_half']}, m={r['m']}, c={r['c_target']}): "
              f"F={r['surv_F']:,} -> N={r['surv_N']:,} -> O={r['surv_O']:,}  "
              f"(N: +{r['pct_NF']:.2f}%; O: +{r['pct_ON']:.2f}% over N, "
              f"+{r['pct_OF']:.2f}% over F)  "
              f"violations: NF={r['soundness_violations_NF']}, "
              f"ON={r['soundness_violations_ON']}, "
              f"OF={r['soundness_violations_OF']}")


if __name__ == '__main__':
    main()
