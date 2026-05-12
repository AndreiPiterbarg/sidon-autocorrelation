"""COARSE-GRID variant O: sparsity-aware tightening of the linear cell-variation
   bound, ported from the FINE grid (_O_bench.py).

DERIVATION
==========
The coarse grid (run_cascade_coarse_v2.py) certifies:
   max_W TV_W(mu) >= c_target  for all mu in cell, where
   cell = {mu = mu* + delta : |delta_i| <= h, Σdelta = 0, mu_i >= 0},
   mu* = k/S, h = 1/(2S).

The current bound uses
   cell_var = max_{|delta|<=h, Σdelta=0} |grad . delta|
            = (1/(2S)) * Σ_{k=0..d/2-1} (g_sorted[d-1-k] - g_sorted[k]).

OBSERVATION (cell shrinks for sparse k):
For j with k_j = 0, the constraint mu_j = delta_j >= 0 forces delta_j IN [0, h]
(only the POSITIVE half of the box).  The tighter LP is:
   cell_var_O = max{ Σ delta_j * grad_j :
                     |delta_j|<=h for j in T = {k_j>0},
                     0 <= delta_j <= h for j in Z = {k_j=0},
                     Σ delta = 0 }    (then reflect for absolute value)

LP STRUCTURE (vertex enum):
For maximizing (grad . delta), at optimum each delta_j sits at a box endpoint,
matched between |P| positives at +h and |N| negatives at -h, |P|=|N| (Σ=0),
plus the rest at 0 (or between vertices if degenerate, but the LP value is
attained at a vertex of the constrained polytope).

  * Positives (delta=+h, contributes +h*grad_j): can be from Z OR T.  We pick
    the largest-grad indices.
  * Negatives (delta=-h, contributes -h*grad_j): MUST be from T (Z forbids -h).
    We pick the smallest-grad T-indices.

For absolute value, also run on -grad (equivalent: pos<->neg swapped, with
negatives still from T but now corresponding to LARGEST grad in T).

Closed-form vertex-enum recipe (for either sign):
  1. Sort T_indices by grad ascending: T_asc.
  2. Sort all indices by grad descending: A_desc.
  3. For k = 1..min(|T|, d//2):
       neg_set = first k of T_asc (smallest grad in T)
       pos_set = first k of A_desc \ neg_set (largest grad over remaining)
       val_k = h * (Σ_{j in pos_set} grad_j - Σ_{j in neg_set} grad_j)
     Record max over k.

Then run again on -grad and take the max.  Final cell_var_O = max value.

SOUNDNESS
=========
Variant O's polytope is a SUBSET of variant F's polytope (same |delta|<=h,
Σ=0 constraints, plus extra delta_j >= 0 for j in Z), so
    cell_var_O <= cell_var.
Equality when Z is empty.  Strict when sparse.  Pruning with the smaller
correction is sound by the same triangle-inequality cert as the baseline.
"""
import os
import sys
import json
import time

import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions


# =====================================================================
# Helper: pair-count prefix sums (same as run_cascade_coarse_v2)
# =====================================================================

@njit(cache=True)
def _build_pair_prefix(d):
    conv_len = 2 * d - 1
    prefix_nk = np.zeros(conv_len + 1, dtype=np.int64)
    prefix_mk = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        nk = min(k + 1, d, conv_len - k)
        mk = np.int64(1) if (k % 2 == 0 and k // 2 < d) else np.int64(0)
        prefix_nk[k + 1] = prefix_nk[k] + np.int64(nk)
        prefix_mk[k + 1] = prefix_mk[k] + mk
    return prefix_nk, prefix_mk


@njit(cache=True, inline='always')
def _quad_corr(s_lo, ell, d, prefix_nk, prefix_mk, inv_4S2):
    hi = s_lo + ell - 1
    N_W = prefix_nk[hi] - prefix_nk[s_lo]
    M_W = prefix_mk[hi] - prefix_mk[s_lo]
    cross_W = N_W - M_W
    d_sq = np.int64(d) * np.int64(d)
    compl_bound = d_sq - N_W
    pair_bound = min(cross_W, compl_bound)
    if pair_bound <= 0:
        return 0.0
    return (2.0 * np.float64(d) / np.float64(ell)
            * np.float64(pair_bound) * inv_4S2)


# =====================================================================
# Helper: one-sided LP (variant O) — closed form via vertex enumeration
# =====================================================================

@njit(cache=True, inline='always')
def _one_sided_lp(grad, k_int, d, h, work_buf):
    """Return cell_var_O = max over both signs of grad of the constrained LP.

    LP: max Σ delta_j * grad_j
        s.t. |delta_j| <= h for j in T = {k_j > 0}
             0 <= delta_j <= h for j in Z = {k_j = 0}
             Σ delta = 0

    work_buf: int64 scratch of length 4*d (we use ints as indices).
    """
    # Best over both signs of objective.
    best = 0.0

    # Iterate sign in {+1, -1} effectively by negating grad.
    for sign_iter in range(2):
        # sign_grad[j] = grad[j] if sign_iter == 0 else -grad[j]
        # We'll work directly without an array: pre-sort each iteration.

        # Build T_indices sorted ASC by signed grad: smallest signed grad first
        # (those become δ=-h candidates; signed grad small => contribute most to
        #  +h*pos_sum - h*neg_sum when in neg_set).
        # Build all_indices sorted DESC by signed grad: largest first.
        # We'll do simple insertion sort into work_buf slots.

        # work_buf layout:
        #   [0..d):       T_asc (T-indices sorted ascending by signed grad)
        #   [d..2d):      A_desc (all indices sorted descending by signed grad)
        #   [2d..3d):     used flag (0/1)
        # Length used: 3*d (we have 4*d available).

        # Fill T_asc (only T members)
        T_count = 0
        for j in range(d):
            if k_int[j] > 0:
                # insertion sort by signed grad ascending
                key = grad[j] if sign_iter == 0 else -grad[j]
                pos = T_count
                # find insert position
                while pos > 0:
                    prev_idx = work_buf[pos - 1]
                    prev_key = grad[prev_idx] if sign_iter == 0 else -grad[prev_idx]
                    if prev_key > key:
                        work_buf[pos] = work_buf[pos - 1]
                        pos -= 1
                    else:
                        break
                work_buf[pos] = j
                T_count += 1

        # Fill A_desc (all d indices sorted descending by signed grad)
        A_off = d
        for j in range(d):
            key = grad[j] if sign_iter == 0 else -grad[j]
            pos = j
            # insertion sort descending: prev_key < key means swap
            while pos > 0:
                prev_idx = work_buf[A_off + pos - 1]
                prev_key = grad[prev_idx] if sign_iter == 0 else -grad[prev_idx]
                if prev_key < key:
                    work_buf[A_off + pos] = work_buf[A_off + pos - 1]
                    pos -= 1
                else:
                    break
            work_buf[A_off + pos] = j

        # Iterate k = 1..min(T_count, d//2)
        k_max = T_count if T_count < (d // 2) else (d // 2)

        # Precompute cumulative neg_sum (over T_asc)
        # neg_sum_k = signed_grad[T_asc[0]] + ... + signed_grad[T_asc[k-1]]
        # Then full LP value at level k:
        #   pos_set = top k of A_desc EXCLUDING neg_set (i.e., excluding T_asc[0..k-1])
        # We must walk A_desc and skip indices that are in neg_set.

        # 'used' marks indices currently in neg_set.
        used_off = 2 * d
        for j in range(d):
            work_buf[used_off + j] = 0

        neg_sum = 0.0
        for k in range(1, k_max + 1):
            new_neg = work_buf[k - 1]  # T_asc[k-1]
            new_neg_signed = grad[new_neg] if sign_iter == 0 else -grad[new_neg]
            neg_sum += new_neg_signed
            work_buf[used_off + new_neg] = 1

            # Walk A_desc, accumulate top-k not in neg_set
            pos_sum = 0.0
            found = 0
            for jj in range(d):
                idx = work_buf[A_off + jj]
                if work_buf[used_off + idx] == 0:
                    pos_signed = grad[idx] if sign_iter == 0 else -grad[idx]
                    pos_sum += pos_signed
                    found += 1
                    if found == k:
                        break
            if found < k:
                break  # not enough indices

            val = h * (pos_sum - neg_sum)
            # val is for max of (signed grad) . delta.
            # When sign_iter == 1, this equals max of -grad . delta = max(-grad.delta).
            # |grad . delta| = max(grad.delta, -grad.delta), so we just take the max.
            if val > best:
                best = val

        # Reset 'used' is unnecessary; we rebuild buffers next sign_iter.
    return best


# =====================================================================
# Baseline pruner: triangle (cell_var + quad_corr)
# =====================================================================

@njit(parallel=True, cache=True)
def prune_coarse_baseline(batch_int, d, S, c_target):
    """Triangle-style baseline: cell_var + quad_corr (same as
       run_cascade_coarse_v2._prune_no_correction)."""
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d
    h = 1.0 / (2.0 * S_d)

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    for b in prange(B):
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
        grad_arr = np.empty(d, dtype=np.float64)

        for ell in range(2, max_ell + 1):
            if pruned:
                break
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
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    # sort grad ascending
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
                    cell_var *= h  # = / (2S)

                    qc = _quad_corr(s_lo, ell, d, prefix_nk, prefix_mk, inv_4S2)

                    if margin - cell_var - qc >= 0.0:
                        pruned = True
                        break

        if pruned:
            survived[b] = False

    return survived


# =====================================================================
# Variant O pruner: one-sided LP (sparsity aware) + same quad_corr
# =====================================================================

@njit(parallel=True, cache=True)
def prune_coarse_O(batch_int, d, S, c_target):
    """Variant O pruner: cell_var_O (sparse-aware LP) + quad_corr."""
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d
    h = 1.0 / (2.0 * S_d)

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    for b in prange(B):
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
        grad_arr = np.empty(d, dtype=np.float64)
        k_local = np.empty(d, dtype=np.int64)
        for j in range(d):
            k_local[j] = np.int64(batch_int[b, j])
        work_buf = np.empty(4 * d, dtype=np.int64)

        for ell in range(2, max_ell + 1):
            if pruned:
                break
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
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    cell_var_O = _one_sided_lp(grad_arr, k_local, d, h, work_buf)

                    # FP-safety: enforce cell_var_O <= cell_var (true mathematically;
                    # FP roundoff in summation can make _one_sided_lp 1-ulp larger).
                    # Compute cell_var (sort and pair extremes) and clip.
                    grad_sorted = np.empty(d, dtype=np.float64)
                    for i in range(d):
                        grad_sorted[i] = grad_arr[i]
                    for i in range(1, d):
                        key = grad_sorted[i]
                        jj = i - 1
                        while jj >= 0 and grad_sorted[jj] > key:
                            grad_sorted[jj + 1] = grad_sorted[jj]
                            jj -= 1
                        grad_sorted[jj + 1] = key
                    cell_var_BL = 0.0
                    for k in range(d // 2):
                        cell_var_BL += grad_sorted[d - 1 - k] - grad_sorted[k]
                    cell_var_BL *= h
                    if cell_var_O > cell_var_BL:
                        cell_var_O = cell_var_BL

                    qc = _quad_corr(s_lo, ell, d, prefix_nk, prefix_mk, inv_4S2)

                    if margin - cell_var_O - qc >= 0.0:
                        pruned = True
                        break

        if pruned:
            survived[b] = False

    return survived


# =====================================================================
# Soundness checker: fine-grid scan over the cell
# =====================================================================

def _max_TV_at_mu(mu, d):
    """Compute max_W TV_W(mu) for a single mu."""
    S_eff = 1.0  # treat mu as already normalized
    # autoconvolution
    conv = np.zeros(2 * d - 1)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]
    best = 0.0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s_lo in range(2 * d - 1 - n_cv + 1):
            ws = conv[s_lo:s_lo + n_cv].sum()
            tv = (2.0 * d / ell) * ws
            if tv > best:
                best = tv
    return best


def _min_max_TV_in_cell(k_int, S, d, n_samples=300, seed=0):
    """Sample n_samples deltas in the cell {|delta|<=h, Σ=0, mu>=0} and return
       min over samples of max_W TV_W(mu* + delta).  Lower bound on the true
       cell minimum (we want this >= c_target for soundness).
    """
    rng = np.random.default_rng(seed)
    h = 1.0 / (2.0 * S)
    mu_star = k_int.astype(np.float64) / S
    best_min = np.inf

    # Always include the centroid
    samples = [np.zeros(d)]

    # Random samples: draw delta uniformly in box, project Σ=0, clip to mu>=0.
    for _ in range(n_samples - 1):
        delta = rng.uniform(-h, h, size=d)
        # subtract mean to enforce Σ = 0
        delta -= delta.mean()
        # if any delta_j < -mu_star_j, clip and re-normalize Σ
        # do a few iterations of project-and-clip
        for _ in range(20):
            delta = np.clip(delta, np.maximum(-mu_star, -h), h)
            mean = delta.mean()
            if abs(mean) < 1e-15:
                break
            delta -= mean
        # Final clip — ensures feasibility (gives slightly smaller cell, but
        # still in the cell)
        delta = np.clip(delta, np.maximum(-mu_star, -h), h)
        samples.append(delta)

    for delta in samples:
        mu = mu_star + delta
        if (mu < -1e-12).any():
            continue
        tv = _max_TV_at_mu(mu, d)
        if tv < best_min:
            best_min = tv

    return best_min


# =====================================================================
# Bench driver
# =====================================================================

def run(d, S, c_target, sound_check=True, n_sound_samples=50, batch_size=200_000,
        verbose=True):
    n_total = count_compositions(d, S)
    if verbose:
        print(f"\n=== d={d}, S={S}, c_target={c_target} ===")
        print(f"     total compositions = {n_total:,}")

    # Warm
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S
    prune_coarse_baseline(warm, d, S, c_target)
    prune_coarse_O(warm, d, S, c_target)

    # Process all compositions
    n_proc = 0
    surv_BL = 0
    surv_O = 0
    n_sparse = 0
    n_sparse_pruned_BL = 0
    n_sparse_pruned_O = 0
    O_wins = []  # (composition, ...) where O prunes but BL doesn't
    n_violations_OB = 0  # O survivors that BL prunes (impossible if O <= BL)

    t0 = time.time()
    for batch in generate_compositions_batched(d, S, batch_size=batch_size):
        n_proc += len(batch)
        sBL = prune_coarse_baseline(batch, d, S, c_target)
        sO = prune_coarse_O(batch, d, S, c_target)
        surv_BL += int(sBL.sum())
        surv_O += int(sO.sum())

        is_sparse = (batch == 0).any(axis=1)
        n_sparse += int(is_sparse.sum())
        n_sparse_pruned_BL += int((~sBL & is_sparse).sum())
        n_sparse_pruned_O += int((~sO & is_sparse).sum())

        # O wins: pruned by O, not by baseline (sBL=True, sO=False)
        wins_mask = sBL & ~sO
        if wins_mask.any():
            for idx in np.where(wins_mask)[0]:
                if len(O_wins) < 2000:
                    O_wins.append(batch[idx].copy())

        # Soundness sanity (compares relative)
        n_violations_OB += int((sO & ~sBL).sum())  # O survives, BL doesn't —
        # this is impossible if O <= BL (both use cell_var bound, but cell_var_O
        # <= cell_var so O is at least as tight, meaning O prunes if BL prunes).

    elapsed = time.time() - t0

    extra = surv_BL - surv_O
    pct_extra = 100.0 * extra / max(1, surv_BL) if surv_BL > 0 else 0.0
    pct_sparse_total = 100.0 * n_sparse / max(1, n_proc)
    pct_sparse_of_pruned_BL = 100.0 * n_sparse_pruned_BL / max(1, n_proc - surv_BL)
    pct_sparse_of_pruned_O = 100.0 * n_sparse_pruned_O / max(1, n_proc - surv_O)

    print(f"     processed:      {n_proc:,}  "
          f"({pct_sparse_total:.2f}% have at least one zero bin)")
    print(f"     baseline survivors: {surv_BL:,}  "
          f"(pruned: {n_proc - surv_BL:,}, "
          f"{pct_sparse_of_pruned_BL:.2f}% of pruned were sparse)")
    print(f"     O        survivors: {surv_O:,}  "
          f"(pruned: {n_proc - surv_O:,}, "
          f"{pct_sparse_of_pruned_O:.2f}% of pruned were sparse)")
    print(f"     O extra prunes: +{extra:,}  ({pct_extra:.4f}% over baseline)")
    print(f"     wall: {elapsed:.2f}s")
    if n_violations_OB > 0:
        print(f"     *** WARN: {n_violations_OB} cells survive O but pruned by BL "
              f"(SHOULD BE 0; O <= BL by polytope inclusion) ***")

    # Soundness check on O-wins
    n_checked = 0
    n_violation = 0
    if sound_check and len(O_wins) > 0:
        rng = np.random.default_rng(123)
        sel = rng.choice(len(O_wins), size=min(n_sound_samples, len(O_wins)),
                         replace=False)
        print(f"     SOUND CHECK on {len(sel)} O-wins (cells pruned by O, not BL)...")
        for s_idx, idx in enumerate(sel):
            k_int = O_wins[idx]
            min_tv = _min_max_TV_in_cell(k_int, S, d, n_samples=400,
                                          seed=int(idx) + 1)
            n_checked += 1
            if min_tv < c_target - 1e-9:
                n_violation += 1
                print(f"     *** VIOLATION: k={k_int.tolist()}, "
                      f"min_max_TV={min_tv:.6f} < c_target={c_target} ***")
        if n_violation == 0:
            print(f"     SOUND CHECK PASSED: 0 violations in {n_checked} samples.")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_processed': n_proc,
        'surv_BL': surv_BL, 'surv_O': surv_O,
        'extra': extra, 'pct_extra': pct_extra,
        'n_sparse_total': n_sparse,
        'pct_sparse_total': pct_sparse_total,
        'n_sparse_pruned_BL': n_sparse_pruned_BL,
        'pct_sparse_of_pruned_BL': pct_sparse_of_pruned_BL,
        'n_sparse_pruned_O': n_sparse_pruned_O,
        'pct_sparse_of_pruned_O': pct_sparse_of_pruned_O,
        'wall_sec': elapsed,
        'sound_n_checked': n_checked,
        'sound_n_violations': n_violation,
        'monotone_violations_OB': n_violations_OB,
        'n_O_wins_total': len(O_wins),
    }


def main():
    configs = [
        (4, 20, 1.20),
        (4, 30, 1.25),
        (6, 15, 1.20),
        (6, 20, 1.25),
        (8, 12, 1.20),
    ]
    results = []
    for (d, S, c) in configs:
        results.append(run(d, S, c))
    out = os.path.join(_dir, '_coarse_O_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 70}\nSummary:")
    for r in results:
        print(f"  (d={r['d']}, S={r['S']}, c={r['c_target']}): "
              f"BL={r['surv_BL']:,} -> O={r['surv_O']:,}, "
              f"+{r['extra']:,} ({r['pct_extra']:.4f}%) "
              f"sparse-frac-of-pruned: BL={r['pct_sparse_of_pruned_BL']:.1f}%, "
              f"O={r['pct_sparse_of_pruned_O']:.1f}%; "
              f"sound: {r['sound_n_violations']}/{r['sound_n_checked']}")


if __name__ == '__main__':
    main()
