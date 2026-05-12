"""Benchmark fast (numba) coarse-F at d=16 to estimate full-enum feasibility.

Uses the existing _prune_coarse from run_cascade.py — numba-JIT'd, F-style
coarse-cascade bound (grid-prune + cell-net cell-cert).

NOT palindromic. Canonical enumeration.
"""
from __future__ import annotations
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_canonical_compositions_batched
from run_cascade import _prune_coarse
import numba
from numba import njit, prange


@njit(parallel=True, cache=True)
def _prune_coarse_count_cell(batch_int, d, S, c_target):
    """Like _prune_coarse but ALSO counts cells with cell_net <= 0.

    Returns (survived, n_neg_net, min_net).
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)
    neg_net_mask = np.zeros(B, dtype=numba.boolean)

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq / (2.0 * d_d)
                                  - eps)

    # Pre-compute prefix_nk and prefix_mk for cross-term bound
    prefix_nk = np.zeros(conv_len + 1, dtype=np.int64)
    prefix_mk = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        nk = min(k + 1, d, conv_len - k)
        mk = np.int64(1) if (k % 2 == 0 and k // 2 < d) else np.int64(0)
        prefix_nk[k + 1] = prefix_nk[k] + np.int64(nk)
        prefix_mk[k + 1] = prefix_mk[k] + mk

    n_threads = numba.config.NUMBA_NUM_THREADS
    min_net_arr = np.full(n_threads, 1e30, dtype=np.float64)
    n_neg_arr = np.zeros(n_threads, dtype=np.int64)

    for b in prange(B):
        tid = numba.get_thread_id()
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        pruned = False
        best_net = -1e30
        grad_arr = np.empty(d, dtype=np.float64)

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                if ws > dyn_it:
                    pruned = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g
                    for i in range(1, d):
                        key = grad_arr[i]
                        jj = i - 1
                        while jj >= 0 and grad_arr[jj] > key:
                            grad_arr[jj + 1] = grad_arr[jj]
                            jj -= 1
                        grad_arr[jj + 1] = key
                    cell_var = 0.0
                    for k in range(d // 2):
                        cell_var += grad_arr[d - 1 - k] - grad_arr[k]
                    cell_var /= (2.0 * S_d)

                    hi_idx = s_lo + ell - 1
                    N_W = prefix_nk[hi_idx] - prefix_nk[s_lo]
                    M_W = prefix_mk[hi_idx] - prefix_mk[s_lo]
                    cross_W = N_W - M_W
                    d_sq = np.int64(d) * np.int64(d)
                    compl_bound = d_sq - N_W
                    pb = min(cross_W, compl_bound)
                    qc = (2.0 * d_d / ell_f) * np.float64(
                        max(pb, np.int64(0))) * inv_4S2

                    net = margin - cell_var - qc
                    if net > best_net:
                        best_net = net

        if pruned:
            survived[b] = False
            if best_net < min_net_arr[tid]:
                min_net_arr[tid] = best_net
            if best_net <= 0.0:
                neg_net_mask[b] = True
                n_neg_arr[tid] += 1

    global_min_net = 1e30
    global_n_neg = np.int64(0)
    for t in range(n_threads):
        if min_net_arr[t] < global_min_net:
            global_min_net = min_net_arr[t]
        global_n_neg += n_neg_arr[t]

    return survived, neg_net_mask, global_n_neg, global_min_net

import math


def total_canonical(d, S):
    """Number of canonical compositions of S into d bins."""
    total = math.comb(S + d - 1, d - 1)
    # Palindromes (= compositions equal to their reverse)
    # For d even: palindrome iff c_i = c_{d-1-i} for i < d/2
    # ⇒ effectively choose c_0..c_{d/2-1} with 2*sum + (c_{d/2-1} if d odd) = S
    # For d even and S even: pal_count = C(S/2 + d/2 - 1, d/2 - 1)
    # For d even and S odd: pal_count = 0
    if d % 2 == 0:
        if S % 2 == 0:
            pal = math.comb(S // 2 + d // 2 - 1, d // 2 - 1)
        else:
            pal = 0
    else:
        pal = math.comb((S - (S % 2 * 0)) // 2 + (d - 1) // 2, (d - 1) // 2)
    return (total + pal) // 2


def bench_d16(S, c_target, batch_size=200_000, max_batches=20):
    d = 16
    print(f"\n=== d=16, S={S}, c_target={c_target} ===", flush=True)
    n_total = total_canonical(d, S)
    print(f"  total canonical compositions: {n_total:,}", flush=True)
    print(f"  warming up JIT...", flush=True)
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S
    t_w = time.time()
    _prune_coarse(warm, d, S, c_target)
    print(f"  JIT warm done in {time.time()-t_w:.1f}s", flush=True)

    n_processed = 0
    n_grid_pruned = 0
    n_cell_certified = 0
    n_grid_surv = 0
    n_cell_uncertain = 0
    min_net_global = np.inf
    t0 = time.time()
    # JIT-warm the count kernel too
    warm2 = np.zeros((1, d), dtype=np.int32)
    warm2[0, 0] = S
    _ = _prune_coarse_count_cell(warm2, d, S, c_target)

    batch_i = 0
    for batch in generate_canonical_compositions_batched(d, S,
                                                            batch_size=batch_size):
        batch_i += 1
        batch_int32 = batch.astype(np.int32)
        t_start = time.time()
        survived, neg_mask, n_neg, min_net = _prune_coarse_count_cell(
            batch_int32, d, S, c_target)
        elapsed = time.time() - t_start
        n_p = int((~survived).sum())
        n_s = int(survived.sum())
        n_neg_batch = int(n_neg)
        n_processed += len(batch)
        n_grid_pruned += n_p
        n_grid_surv += n_s
        n_cell_uncertain += n_neg_batch
        if min_net < min_net_global:
            min_net_global = min_net
        rate = len(batch) / max(elapsed, 1e-9)
        total_elapsed = time.time() - t0
        eta = (n_total - n_processed) / max(rate, 1)
        print(f"  batch {batch_i}: processed={n_processed:>11,}/{n_total:,} "
              f"({100*n_processed/n_total:.2f}%)  grid_pruned={n_p:>9,}  "
              f"grid_surv={n_s:>9,}  cell_uncertain={n_neg_batch:>7,}  "
              f"min_net={min_net:.4f}  rate={rate/1e6:.2f}M/s  ETA={eta:.1f}s",
              flush=True)
        if batch_i >= max_batches:
            print(f"  (capped at {max_batches} batches for time)", flush=True)
            break

    total_t = time.time() - t0
    avg_rate = n_processed / max(total_t, 1e-9)
    full_eta = n_total / max(avg_rate, 1)
    print(f"\n  After {batch_i} batches:", flush=True)
    print(f"    processed:       {n_processed:,}", flush=True)
    print(f"    grid-pruned:     {n_grid_pruned:,} "
          f"({100*n_grid_pruned/n_processed:.2f}%)", flush=True)
    print(f"    grid-survivors:  {n_grid_surv:,} "
          f"({100*n_grid_surv/n_processed:.4f}%)", flush=True)
    print(f"    cell-uncertain:  {n_cell_uncertain:,} "
          f"({100*n_cell_uncertain/n_processed:.4f}%)  "
          f"(grid-pruned but net<=0; need v2 L tier)", flush=True)
    print(f"    min_cert_net (over grid-pruned cells): {min_net_global:.6f}", flush=True)
    if min_net_global > 0:
        print(f"    *** All grid-pruned cells are CELL-CERTIFIED ***", flush=True)
    else:
        print(f"    *** Some grid-pruned cells need refinement (min_net <= 0) ***", flush=True)
    print(f"    elapsed:         {total_t:.2f}s", flush=True)
    print(f"    avg rate:        {avg_rate/1e6:.2f}M/s", flush=True)
    print(f"    full enum ETA:   {full_eta:.0f}s = {full_eta/60:.1f} min", flush=True)
    return {'d': 16, 'S': S, 'c_target': c_target,
             'processed': n_processed, 'total': n_total,
             'grid_pruned': n_grid_pruned, 'grid_surv': n_grid_surv,
             'cell_uncertain': n_cell_uncertain,
             'min_net': min_net_global, 'elapsed_s': total_t,
             'avg_rate_per_s': avg_rate, 'full_eta_s': full_eta}


if __name__ == '__main__':
    results = []
    # Short test: just S=16 with full ETA estimate
    r = bench_d16(16, c_target=1.28, batch_size=200_000, max_batches=10)
    results.append(r)

    print(f"\n=== SUMMARY ===", flush=True)
    print(f"{'S':>4}  {'total':>15}  {'rate':>10}  {'full ETA':>12}  "
          f"{'cell-uncert%':>12}  {'min_net':>8}", flush=True)
    for r in results:
        print(f"{r['S']:>4}  {r['total']:>15,}  "
              f"{r['avg_rate_per_s']/1e6:>8.2f}M/s  "
              f"{r['full_eta_s']:>10.1f}s  "
              f"{100*r['cell_uncertain']/r['processed']:>11.4f}%  "
              f"{r['min_net']:>+8.4f}", flush=True)
