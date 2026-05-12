"""Test Idea 2: Auxiliary grid re-check for additional pruning.

For each survivor at grid (d, m), re-evaluate the same physical mass
distribution at a different grid resolution m'.  If it's pruned at m',
the distribution is dead.

The key insight: a step function g at grid m represents a specific
piecewise-constant function.  This same function can be expressed at
grid m' by rescaling: c'_i = round(c_i * m'/m).  The total should be
S' = 4*n*m' (or 2*d*m' for the fine grid).

For integer multiples m' = k*m: c'_i = k*c_i exactly, S' = k*S.
The correction becomes 2/m' + 1/m'^2 < 2/m + 1/m^2, giving a tighter
threshold.

This is sound because the step function g is the same physical function
regardless of which grid we use to analyze it.  If the m'-grid analysis
proves max(f*f) >= c_target for all continuous functions with the same
bin masses, the distribution is pruned.
"""
import sys
import os
import time
import numpy as np
import numba
from numba import njit, prange

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from importlib import import_module
_cs = import_module('cloninger-steinerberger.pruning')
_rc = import_module('cloninger-steinerberger.cpu.run_cascade')

correction = _cs.correction
count_compositions = _cs.count_compositions
_canonical_mask = _cs._canonical_mask
_prune_dynamic = _rc._prune_dynamic
_prune_dynamic_int32 = _rc._prune_dynamic_int32
_l0_bnb_run = _rc._l0_bnb_run
_canonicalize_inplace = _rc._canonicalize_inplace
_fast_dedup = _rc._fast_dedup
process_parent_fused = _rc.process_parent_fused
run_level0 = _rc.run_level0


@njit(parallel=True, cache=True)
def _recheck_at_m_prime(batch_int, n_half, m_orig, m_prime, c_target):
    """Re-evaluate survivors at grid m' and return mask of those still surviving.

    For each row c_i (integer masses at grid m with sum S = 4*n*m_orig),
    compute c'_i = round(c_i * m_prime / m_orig).  Adjust so sum = S' = 4*n*m_prime.
    Then prune using the m'-grid correction (2/m' + 1/m'^2).

    Returns boolean mask: True = survived (not pruned at m' either).
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    survived = np.ones(B, dtype=numba.boolean)

    S_orig = np.int64(0)
    for i in range(d):
        S_orig += np.int64(batch_int[0, i])  # all rows have same sum
    S_prime = np.int64(round(float(S_orig) * float(m_prime) / float(m_orig)))

    # m'-grid parameters
    m_d = np.float64(m_prime)
    four_n = 4.0 * np.float64(n_half)
    n_half_d = np.float64(n_half)
    cs_base_m2 = c_target * m_d * m_d
    eps_margin = 1e-9 * m_d * m_d
    conv_len = 2 * d - 1
    d_m1 = d - 1
    max_ell = 2 * d
    ratio = np.float64(m_prime) / np.float64(m_orig)

    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    for b in prange(B):
        # Rescale to m'-grid
        c_prime = np.empty(d, dtype=np.int32)
        running_sum = np.int64(0)
        for i in range(d - 1):
            c_prime[i] = np.int32(round(np.float64(batch_int[b, i]) * ratio))
            if c_prime[i] < 0:
                c_prime[i] = 0
            running_sum += np.int64(c_prime[i])
        # Force last bin to make sum exact
        c_prime[d - 1] = np.int32(S_prime - running_sum)
        if c_prime[d - 1] < 0:
            # Rounding error made sum too large; skip this entry
            continue

        # Compute autoconvolution at m'-grid
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(c_prime[i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(c_prime[j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        # Prefix sum for W_int
        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(c_prime[i])

        # Window scan with W-refined threshold
        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            sc = scale_arr[ell]
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                lo_b = s_lo - d_m1
                if lo_b < 0:
                    lo_b = 0
                hi_b = s_lo + ell - 2
                if hi_b > d_m1:
                    hi_b = d_m1
                W_int = prefix_c[hi_b + 1] - prefix_c[lo_b]
                cw = 1.0 + np.float64(W_int) / (2.0 * n_half_d)
                dyn_it = np.int64((cs_base_m2 + cw + eps_margin) * sc)
                if ws > dyn_it:
                    pruned = True
                    break

        if pruned:
            survived[b] = False

    return survived


def recheck_survivors(survivors, n_half, m, c_target, m_primes):
    """Apply auxiliary grid re-check at multiple m' values.

    Returns mask of survivors that pass ALL m' checks.
    """
    survived = np.ones(len(survivors), dtype=bool)
    for m_prime in m_primes:
        if m_prime == m:
            continue
        mask = _recheck_at_m_prime(survivors[survived], n_half, m, m_prime,
                                    c_target)
        # Map back to original indices
        idx_survived = np.where(survived)[0]
        for i in range(len(mask)):
            if not mask[i]:
                survived[idx_survived[i]] = False
    return survived


def run_comparison(d0, m, c_target, m_primes, max_l1_parents=50, max_l2_parents=20):
    """Run baseline vs aux-grid-recheck and compare L0, L1, L2."""
    d = d0
    n_half = d / 2.0
    S = int(2 * d * m)

    print(f"{'='*70}")
    print(f"Config: d0={d0}, m={m}, c_target={c_target}, S={S}")
    print(f"Auxiliary grids: m' = {m_primes}")
    print(f"{'='*70}")

    # --- L0: Baseline ---
    t0 = time.time()
    surv_arr, n_surv_raw, n_tested = _l0_bnb_run(
        d, S, n_half, m, c_target, False)
    if len(surv_arr) > 0:
        mask = _canonical_mask(surv_arr)
        surv_arr = surv_arr[mask]
    t_base = time.time() - t0
    surv_base = surv_arr.copy()

    # --- L0: Aux grid recheck ---
    t0 = time.time()
    if len(surv_arr) > 0:
        aux_mask = recheck_survivors(surv_arr, n_half, m, c_target, m_primes)
        surv_aux = surv_arr[aux_mask]
    else:
        surv_aux = surv_arr
        aux_mask = np.array([], dtype=bool)
    t_aux = time.time() - t0

    n_pruned_by_aux = len(surv_base) - len(surv_aux)
    print(f"\n--- L0 (d={d}) ---")
    print(f"  Baseline:     {len(surv_base):,} survivors ({t_base:.3f}s)")
    print(f"  + Aux recheck: {len(surv_aux):,} survivors ({t_aux:.3f}s)")
    print(f"  Additional pruned by aux grid: {n_pruned_by_aux}")

    if n_pruned_by_aux > 0:
        pruned_rows = surv_base[~aux_mask]
        print(f"  Pruned distributions: {[list(r) for r in pruned_rows[:10]]}")

    # Check each m' individually
    if len(surv_base) > 0:
        for mp in m_primes:
            mask_mp = _recheck_at_m_prime(surv_base, n_half, m, mp, c_target)
            n_killed = int(np.sum(~mask_mp))
            corr_mp = 2.0/mp + 1.0/(mp*mp)
            print(f"    m'={mp:3d}: kills {n_killed}/{len(surv_base)} "
                  f"(correction {corr_mp:.4f} vs {2.0/m + 1.0/(m*m):.4f})")

    # --- L1: expand L0 survivors ---
    n_half_child = d
    d_child = 2 * d

    def expand_to_l1(l0_survivors, label, max_parents):
        n_parents = min(len(l0_survivors), max_parents)
        total_children = 0
        total_l1_surv = 0
        l1_all = []
        t0 = time.time()
        for i in range(n_parents):
            surv, tc = process_parent_fused(
                l0_survivors[i], m, c_target, n_half_child)
            total_children += tc
            total_l1_surv += len(surv)
            if len(surv) > 0:
                l1_all.append(surv)
        elapsed = time.time() - t0
        print(f"  {label}: {n_parents} parents -> {total_children:,} children "
              f"-> {total_l1_surv:,} L1 survivors ({elapsed:.2f}s)")
        if l1_all:
            return np.vstack(l1_all), total_l1_surv, total_children
        return np.empty((0, d_child), dtype=np.int32), 0, total_children

    print(f"\n--- L0->L1 (d={d} -> d={d_child}) ---")
    l1_base, l1_s_b, l1_c_b = expand_to_l1(surv_base, "Baseline ", max_l1_parents)
    l1_aux, l1_s_a, l1_c_a = expand_to_l1(surv_aux, "Aux-grid ", max_l1_parents)
    if l1_c_b > 0:
        print(f"  L1 children saved: {l1_c_b - l1_c_a:,}")
        print(f"  L1 survivors saved: {l1_s_b - l1_s_a:,}")

    # --- Aux recheck on L1 survivors ---
    if len(l1_base) > 0:
        n_half_l1 = float(d)  # n_half at d_child level
        # Scale m_primes for L1
        print(f"\n--- Aux recheck on L1 survivors (d={d_child}) ---")
        for mp in m_primes:
            mask_l1 = _recheck_at_m_prime(l1_base, n_half_l1, m, mp, c_target)
            n_killed = int(np.sum(~mask_l1))
            print(f"    m'={mp:3d}: kills {n_killed}/{len(l1_base)} L1 survivors")

        # Combined: baseline L1 survivors that survive aux recheck
        best_mask = np.ones(len(l1_base), dtype=bool)
        for mp in m_primes:
            mask_l1 = _recheck_at_m_prime(l1_base[best_mask], n_half_l1, m, mp, c_target)
            idx = np.where(best_mask)[0]
            for i in range(len(mask_l1)):
                if not mask_l1[i]:
                    best_mask[idx[i]] = False
        l1_after_aux = l1_base[best_mask]
        print(f"  Combined aux recheck: {len(l1_base)} -> {len(l1_after_aux)} "
              f"({len(l1_base) - len(l1_after_aux)} pruned)")

    # --- L2: expand sample of L1 survivors ---
    if max_l2_parents > 0 and len(l1_base) > 0:
        n_half_l2 = 2 * d
        d_child_l2 = 4 * d

        def expand_to_l2(l1_survivors, label, max_parents):
            n_parents = min(len(l1_survivors), max_parents)
            total_children = 0
            total_l2_surv = 0
            t0 = time.time()
            for i in range(n_parents):
                surv, tc = process_parent_fused(
                    l1_survivors[i], m, c_target, 2 * d)
                total_children += tc
                total_l2_surv += len(surv)
            elapsed = time.time() - t0
            print(f"  {label}: {n_parents} L1 parents -> {total_children:,} "
                  f"children -> {total_l2_surv:,} L2 survivors ({elapsed:.2f}s)")
            return total_l2_surv, total_children

        print(f"\n--- L1->L2 (d={d_child} -> d={d_child_l2}) ---")
        l2_s_b, l2_c_b = expand_to_l2(l1_base, "Baseline ", max_l2_parents)
        l2_s_a, l2_c_a = expand_to_l2(l1_aux, "From aux-L0", max_l2_parents)
        if l2_c_b > 0:
            print(f"  L2 children saved: {l2_c_b - l2_c_a:,}")
            print(f"  L2 survivors saved: {l2_s_b - l2_s_a:,}")


if __name__ == '__main__':
    print("Warming up JIT...")
    dummy = np.zeros((1, 2), dtype=np.int32)
    _recheck_at_m_prime(dummy, 1.0, 7, 14, 1.28)
    _l0_bnb_run(2, 28, 1.0, 7, 1.28, False)
    print("JIT warm, starting tests.\n")

    configs = [
        # (d0, m, c_target, m_primes, max_l1, max_l2)
        (2, 7, 1.28, [14, 21, 28, 35, 49, 70, 140], 50, 20),
        (2, 7, 1.40, [14, 21, 28, 35, 49, 70, 140], 50, 20),
        (2, 20, 1.28, [40, 60, 100, 140, 200], 50, 10),
        (2, 20, 1.40, [40, 60, 100, 140, 200], 50, 10),
        (3, 7, 1.28, [14, 21, 35, 70], 20, 5),
        (4, 7, 1.28, [14, 21, 35, 70], 10, 3),
    ]

    for d0, m, c_target, m_primes, ml1, ml2 in configs:
        run_comparison(d0, m, c_target, m_primes, ml1, ml2)
        print()
