"""Solver routines: run_single_level, find_best_bound, find_best_bound_direct.

Contains the main entry points and the Numba JIT kernels for
fused generation + pruning + test value computation.

Optimizations:
  - Fused Numba kernels for run_single_level (eliminates Python overhead)
  - Window-bound loop tightening: ell=4 pair-sum bound restricts c1 range,
    ell=3 right-window bound restricts c2 range (M4: non-binding cells)
  - Interleaved c0 ordering for load-balanced parallelism across threads
"""
import numpy as np
import numba
import time
import math

from pruning import correction, asymmetry_threshold, count_compositions, asymmetry_prune_mask
from compositions import generate_canonical_compositions_batched
from test_values import compute_test_values_batch, compute_test_value_single, _test_values_jit


# =====================================================================
# Work distribution helper
# =====================================================================

@numba.njit(cache=True)
def _build_interleaved_order(n):
    """Build interleaved index ordering for load-balanced prange.

    Maps sequential indices to c0 values such that consecutive indices
    alternate between heavy (small c0) and light (large c0) work items.
    With static prange scheduling, each thread's contiguous chunk gets
    a balanced mix of heavy and light work.

    Pattern: 0, n-1, 1, n-2, 2, n-3, ...
    """
    order = np.empty(n, dtype=np.int32)
    lo, hi = 0, n - 1
    idx = 0
    while lo <= hi:
        order[idx] = lo
        idx += 1
        if lo < hi:
            order[idx] = hi
            idx += 1
        lo += 1
        hi -= 1
    return order


# =====================================================================
# Fused Numba kernels for find_best_bound_direct (single-pass minimum)
# =====================================================================

@numba.njit(parallel=True, cache=True)
def _find_min_eff_d4(c0_order, S, n_half, inv_m, margin, corr, init_min_eff):
    """All-in-one: generate canonical d=4, asymmetry filter, test values, find min.

    M4 optimization: ell=4 pair-sum bound tightens c1 loop range.
    Also uses ell=3 right-window bound to set c2 lower bound.
    Uses interleaved c0 ordering for balanced parallel load distribution.
    """
    d = 4
    conv_len = 2 * d - 1
    total_mass = float(S)
    n_c0 = len(c0_order)
    scale = 4.0 * n_half * inv_m   # c -> a conversion factor (4n/m)

    thread_mins = np.full(n_c0, 1e30, dtype=np.float64)
    thread_cfg = np.zeros((n_c0, d), dtype=np.int32)

    for c0_idx in numba.prange(n_c0):
        c0 = c0_order[c0_idx]

        local_min = init_min_eff
        r0 = S - c0
        c1_max_canon = r0 - c0
        local_cfg = np.zeros(d, dtype=np.int32)
        conv = np.zeros(conv_len, dtype=np.float64)

        for c1 in range(c1_max_canon + 1):
            r1 = r0 - c1

            early_thresh = local_min + corr
            norm_ell_d = 4.0 * n_half * d
            pair_left = float(c0 + c1)
            pair_right = float(r1)
            if pair_left * pair_left * scale * scale / norm_ell_d > early_thresh:
                continue
            if pair_right * pair_right * scale * scale / norm_ell_d > early_thresh:
                continue

            c2_max = r1 - c0

            left_frac = float(c0 + c1) / total_mass
            if left_frac > 0.5:
                dom = left_frac
            else:
                dom = 1.0 - left_frac
            asym_base = dom - margin
            if asym_base < 0.0:
                asym_base = 0.0
            asym_val = 2.0 * asym_base * asym_base
            if asym_val >= local_min:
                continue

            norm_ell3 = 4.0 * n_half * 3
            r1_sq = float(r1) * float(r1)
            ell3_cutoff_sq = r1_sq - early_thresh * norm_ell3 / (scale * scale)
            if ell3_cutoff_sq > 0.0:
                c2_lo_ell3 = int(math.sqrt(ell3_cutoff_sq)) + 1
            else:
                c2_lo_ell3 = 0
            c2_start = max(0, c2_lo_ell3)

            inv_ell2 = 1.0 / (4.0 * n_half * 2)
            for c2 in range(c2_start, c2_max + 1):
                c3 = r1 - c2
                if c0 == c3 and c1 > c2:
                    continue

                early_thresh = local_min + corr
                max_c = c3
                if c1 > max_c:
                    max_c = c1
                if c2 > max_c:
                    max_c = c2
                max_a = max_c * scale
                if max_a * max_a * inv_ell2 > early_thresh:
                    continue

                a0 = c0 * scale
                a1 = c1 * scale
                a2 = c2 * scale
                a3 = c3 * scale
                conv[0] = a0 * a0
                conv[1] = 2.0 * a0 * a1
                conv[2] = a1 * a1 + 2.0 * a0 * a2
                conv[3] = 2.0 * (a0 * a3 + a1 * a2)
                conv[4] = a2 * a2 + 2.0 * a1 * a3
                conv[5] = 2.0 * a2 * a3
                conv[6] = a3 * a3

                for k in range(1, conv_len):
                    conv[k] += conv[k - 1]

                best = 0.0
                done = False
                for ell in range(2, d + 1):
                    if done:
                        break
                    n_cv = ell - 1
                    inv_norm = 1.0 / (4.0 * n_half * ell)
                    for s_lo in range(conv_len - n_cv + 1):
                        s_hi = s_lo + n_cv - 1
                        ws = conv[s_hi]
                        if s_lo > 0:
                            ws -= conv[s_lo - 1]
                        tv = ws * inv_norm
                        if tv > best:
                            best = tv
                            if best > early_thresh:
                                done = True
                                break

                eff = best - corr
                if asym_val > eff:
                    eff = asym_val
                if eff < local_min:
                    local_min = eff
                    local_cfg[0] = c0
                    local_cfg[1] = c1
                    local_cfg[2] = c2
                    local_cfg[3] = c3

        thread_mins[c0_idx] = local_min
        for i in range(d):
            thread_cfg[c0_idx, i] = local_cfg[i]

    return thread_mins, thread_cfg


@numba.njit(parallel=True, cache=True)
def _find_min_eff_d6(c0_order, S, n_half, inv_m, margin, corr, init_min_eff):
    """All-in-one for d=6: canonical gen + asymmetry + test values.

    M4 optimization: ell=6 three-bin sum bound at (c0,c1,c2) level.
    Uses interleaved c0 ordering for balanced parallel load distribution.
    """
    d = 6
    conv_len = 2 * d - 1
    total_mass = float(S)
    n_c0 = len(c0_order)
    scale = 4.0 * n_half * inv_m   # c -> a conversion factor (4n/m)

    thread_mins = np.full(n_c0, 1e30, dtype=np.float64)
    thread_cfg = np.zeros((n_c0, d), dtype=np.int32)

    for c0_idx in numba.prange(n_c0):
        c0 = c0_order[c0_idx]
        local_min = init_min_eff
        r0 = S - c0
        local_cfg = np.zeros(d, dtype=np.int32)
        conv = np.zeros(conv_len, dtype=np.float64)

        for c1 in range(r0 + 1):
            r1 = r0 - c1
            for c2 in range(r1 + 1):
                r2 = r1 - c2

                # M4: ell=6 three-bin sum bound
                early_thresh = local_min + corr
                norm_ell_d = 4.0 * n_half * d
                left3 = float(c0 + c1 + c2)
                right3 = float(r2)
                if left3 * left3 * scale * scale / norm_ell_d > early_thresh:
                    continue
                if right3 * right3 * scale * scale / norm_ell_d > early_thresh:
                    continue

                # Asymmetry
                left_frac2 = left3 / total_mass
                if left_frac2 > 0.5:
                    dom2 = left_frac2
                else:
                    dom2 = 1.0 - left_frac2
                asym_base2 = dom2 - margin
                if asym_base2 < 0.0:
                    asym_base2 = 0.0
                asym_val = 2.0 * asym_base2 * asym_base2
                if asym_val >= local_min:
                    continue

                for c3 in range(r2 + 1):
                    r3 = r2 - c3
                    c4_max = r3 - c0
                    if c4_max < 0:
                        break
                    inv_ell2 = 1.0 / (4.0 * n_half * 2)
                    for c4 in range(c4_max + 1):
                        c5 = r3 - c4
                        if c0 == c5:
                            if c1 > c4:
                                continue
                            if c1 == c4 and c2 > c3:
                                continue

                        early_thresh = local_min + corr
                        max_c = c5
                        if c1 > max_c:
                            max_c = c1
                        if c2 > max_c:
                            max_c = c2
                        if c3 > max_c:
                            max_c = c3
                        if c4 > max_c:
                            max_c = c4
                        max_a_val = max_c * scale
                        if max_a_val * max_a_val * inv_ell2 > early_thresh:
                            continue

                        a0 = c0 * scale
                        a1 = c1 * scale
                        a2 = c2 * scale
                        a3 = c3 * scale
                        a4 = c4 * scale
                        a5 = c5 * scale
                        conv[0] = a0 * a0
                        conv[1] = 2.0 * a0 * a1
                        conv[2] = 2.0 * a0 * a2 + a1 * a1
                        conv[3] = 2.0 * (a0 * a3 + a1 * a2)
                        conv[4] = 2.0 * (a0 * a4 + a1 * a3) + a2 * a2
                        conv[5] = 2.0 * (a0 * a5 + a1 * a4 + a2 * a3)
                        conv[6] = 2.0 * (a1 * a5 + a2 * a4) + a3 * a3
                        conv[7] = 2.0 * (a2 * a5 + a3 * a4)
                        conv[8] = 2.0 * a3 * a5 + a4 * a4
                        conv[9] = 2.0 * a4 * a5
                        conv[10] = a5 * a5

                        for k in range(1, conv_len):
                            conv[k] += conv[k - 1]

                        best = 0.0
                        early_thresh = local_min + corr
                        done = False
                        for ell in range(2, d + 1):
                            if done:
                                break
                            n_cv = ell - 1
                            inv_norm = 1.0 / (4.0 * n_half * ell)
                            for s_lo in range(conv_len - n_cv + 1):
                                s_hi = s_lo + n_cv - 1
                                ws = conv[s_hi]
                                if s_lo > 0:
                                    ws -= conv[s_lo - 1]
                                tv = ws * inv_norm
                                if tv > best:
                                    best = tv
                                    if best > early_thresh:
                                        done = True
                                        break

                        eff = best - corr
                        if asym_val > eff:
                            eff = asym_val
                        if eff < local_min:
                            local_min = eff
                            local_cfg[0] = c0
                            local_cfg[1] = c1
                            local_cfg[2] = c2
                            local_cfg[3] = c3
                            local_cfg[4] = c4
                            local_cfg[5] = c5

        thread_mins[c0_idx] = local_min
        for i in range(d):
            thread_cfg[c0_idx, i] = local_cfg[i]

    return thread_mins, thread_cfg


# =====================================================================
# Generic fused Numba kernel for any d (find_best_bound_direct)
# =====================================================================

@numba.njit(parallel=True, cache=True)
def _find_min_eff_generic(c0_order, S, d, n_half, inv_m, margin, corr, init_min_eff):
    """Generic fused kernel for any d: canonical gen + pruning + test values.

    Uses a state-machine to simulate d-2 nested loops for composition
    generation, with pruning at each level:
      - Left partial sum ell=d window bound
      - Right remaining mass ell=d window bound
      - Remaining mass < c0 early termination (canonical)
      - Max element ell=2 window bound
      - Asymmetry at the halfway point
      - Early exit in test value computation
    """
    conv_len = 2 * d - 1
    total_mass = float(S)
    n_c0 = len(c0_order)
    half_d = d // 2
    scale = 4.0 * n_half * inv_m   # c -> a conversion factor (4n/m)
    norm_d_inv = scale * scale / (4.0 * n_half * d)
    inv_ell2 = 1.0 / (4.0 * n_half * 2)

    thread_mins = np.full(n_c0, 1e30, dtype=np.float64)
    thread_cfg = np.zeros((n_c0, d), dtype=np.int32)

    for c0_idx in numba.prange(n_c0):
        c0 = c0_order[c0_idx]
        local_min = init_min_eff
        local_cfg = np.zeros(d, dtype=np.int32)
        c = np.zeros(d, dtype=np.int32)
        rem = np.zeros(d, dtype=np.int32)
        conv = np.zeros(conv_len, dtype=np.float64)

        c[0] = c0

        # --- d=2 special case ---
        if d == 2:
            c[1] = S - c0
            if c0 <= c[1]:
                a0 = c0 * scale
                a1 = c[1] * scale
                # conv: [a0^2, 2*a0*a1, a1^2], then prefix sum
                conv[0] = a0 * a0
                conv[1] = conv[0] + 2.0 * a0 * a1
                conv[2] = conv[1] + a1 * a1
                # Only ell=2: 1 conv entry, 3 positions
                inv_norm = 1.0 / (4.0 * n_half * 2)
                best = 0.0
                for s_lo in range(3):
                    ws = conv[s_lo]
                    if s_lo > 0:
                        ws -= conv[s_lo - 1]
                    tv = ws * inv_norm
                    if tv > best:
                        best = tv
                eff = best - corr
                left_frac = float(c0) / total_mass
                if left_frac > 0.5:
                    dom = left_frac
                else:
                    dom = 1.0 - left_frac
                asym_base = dom - margin
                if asym_base < 0.0:
                    asym_base = 0.0
                asym_val = 2.0 * asym_base * asym_base
                if asym_val > eff:
                    eff = asym_val
                if eff < local_min:
                    local_min = eff
                    local_cfg[0] = c0
                    local_cfg[1] = c[1]
            thread_mins[c0_idx] = local_min
            for i in range(d):
                thread_cfg[c0_idx, i] = local_cfg[i]
            continue

        # --- General d >= 3: state machine over c[1..d-2] ---
        # rem[p] = remaining mass for c[p] through c[d-1]
        rem[1] = S - c0
        pos = 1
        c[1] = 0

        while pos >= 1:
            if pos == d - 2:
                # Innermost free position: iterate c[d-2], c[d-1] determined
                avail = rem[pos]
                c_max = avail - c0  # canonical: c[d-1] >= c[0]

                while c[pos] <= c_max:
                    c[d - 1] = avail - c[pos]

                    # Canonical check: c <= rev(c) lexicographically
                    # c[0] <= c[d-1] enforced by loop bound; check inner pairs
                    is_canon = True
                    for i in range(half_d):
                        j = d - 1 - i
                        if i >= j:
                            break
                        if c[i] < c[j]:
                            break
                        if c[i] > c[j]:
                            is_canon = False
                            break
                    if not is_canon:
                        c[pos] += 1
                        continue

                    # ell=2 max element pruning
                    early_thresh = local_min + corr
                    max_c = c[0]
                    for i in range(1, d):
                        if c[i] > max_c:
                            max_c = c[i]
                    max_a = max_c * scale
                    if max_a * max_a * inv_ell2 > early_thresh:
                        c[pos] += 1
                        continue

                    # Asymmetry check
                    left_sum = 0.0
                    for i in range(half_d):
                        left_sum += c[i]
                    left_frac = left_sum / total_mass
                    if left_frac > 0.5:
                        dom = left_frac
                    else:
                        dom = 1.0 - left_frac
                    asym_base = dom - margin
                    if asym_base < 0.0:
                        asym_base = 0.0
                    asym_val = 2.0 * asym_base * asym_base
                    if asym_val >= local_min:
                        c[pos] += 1
                        continue

                    # Compute convolution a*a (symmetric: half the work)
                    for k in range(conv_len):
                        conv[k] = 0.0
                    for ii in range(d):
                        ai = c[ii] * scale
                        conv[2 * ii] += ai * ai
                        for jj in range(ii + 1, d):
                            conv[ii + jj] += 2.0 * ai * (c[jj] * scale)
                    # Prefix sum
                    for k in range(1, conv_len):
                        conv[k] += conv[k - 1]

                    # Find best windowed test value
                    best = 0.0
                    early_thresh = local_min + corr
                    done = False
                    for ell in range(2, d + 1):
                        if done:
                            break
                        n_cv = ell - 1
                        inv_norm = 1.0 / (4.0 * n_half * ell)
                        for s_lo in range(conv_len - n_cv + 1):
                            s_hi = s_lo + n_cv - 1
                            ws = conv[s_hi]
                            if s_lo > 0:
                                ws -= conv[s_lo - 1]
                            tv = ws * inv_norm
                            if tv > best:
                                best = tv
                                if best > early_thresh:
                                    done = True
                                    break

                    eff = best - corr
                    if asym_val > eff:
                        eff = asym_val
                    if eff < local_min:
                        local_min = eff
                        for i in range(d):
                            local_cfg[i] = c[i]

                    c[pos] += 1

                # Backtrack from innermost
                pos -= 1
                if pos >= 1:
                    c[pos] += 1

            elif c[pos] > rem[pos]:
                # Exceeded available mass, backtrack
                pos -= 1
                if pos >= 1:
                    c[pos] += 1

            else:
                # Intermediate position: pruning checks, then advance
                early_thresh = local_min + corr
                prune = False

                # Canonical: remaining mass must allow c[d-1] >= c[0]
                next_rem = rem[pos] - c[pos]
                if next_rem < c0:
                    # All larger c[pos] values also fail; force backtrack
                    c[pos] = rem[pos] + 1
                    continue

                # Left partial sum ell=d window bound
                if not prune:
                    left_partial = 0
                    for i in range(pos + 1):
                        left_partial += c[i]
                    lp = float(left_partial)
                    if lp * lp * norm_d_inv > early_thresh:
                        prune = True

                # Right remaining ell=d window bound
                if not prune:
                    rr = float(next_rem)
                    if rr * rr * norm_d_inv > early_thresh:
                        prune = True

                # Max element so far ell=2 bound
                if not prune:
                    max_c = 0
                    for i in range(pos + 1):
                        if c[i] > max_c:
                            max_c = c[i]
                    max_a = max_c * scale
                    if max_a * max_a * inv_ell2 > early_thresh:
                        prune = True

                # Asymmetry at halfway point
                if not prune and pos == half_d - 1:
                    left_frac = float(left_partial) / total_mass
                    if left_frac > 0.5:
                        dom = left_frac
                    else:
                        dom = 1.0 - left_frac
                    asym_base = dom - margin
                    if asym_base < 0.0:
                        asym_base = 0.0
                    asym_val = 2.0 * asym_base * asym_base
                    if asym_val >= local_min:
                        prune = True

                if prune:
                    c[pos] += 1
                else:
                    rem[pos + 1] = next_rem
                    pos += 1
                    c[pos] = 0

        thread_mins[c0_idx] = local_min
        for i in range(d):
            thread_cfg[c0_idx, i] = local_cfg[i]

    return thread_mins, thread_cfg


# =====================================================================
# Generic fused Numba kernel for any d (run_single_level)
# =====================================================================

@numba.njit(parallel=True, cache=True)
def _prove_target_generic(c0_order, S, d, n_half, inv_m, margin, prune_target, fp_margin):
    """Generic fused kernel for any d: prove all compositions exceed target.

    Same state-machine and pruning as _find_min_eff_generic, but checks
    against a fixed threshold instead of tracking a running minimum.
    """
    conv_len = 2 * d - 1
    total_mass = float(S)
    n_c0 = len(c0_order)
    half_d = d // 2
    thresh = prune_target + fp_margin
    scale = 4.0 * n_half * inv_m   # c -> a conversion factor (4n/m)
    norm_d_inv = scale * scale / (4.0 * n_half * d)
    inv_ell2 = 1.0 / (4.0 * n_half * 2)
    # Asymmetry bound applies directly to c (not through discretization),
    # so compare against c_target, not prune_target.
    c_target = prune_target - 2.0 * inv_m - inv_m * inv_m
    inv_m_sq = inv_m * inv_m

    thread_survivors = np.zeros(n_c0, dtype=np.int64)
    thread_asym = np.zeros(n_c0, dtype=np.int64)
    thread_test = np.zeros(n_c0, dtype=np.int64)
    thread_min_tv = np.full(n_c0, 1e30, dtype=np.float64)
    thread_min_cfg = np.zeros((n_c0, d), dtype=np.int32)

    for c0_idx in numba.prange(n_c0):
        c0 = c0_order[c0_idx]
        local_surv = np.int64(0)
        local_asym = np.int64(0)
        local_test = np.int64(0)
        local_min_tv = 1e30
        local_min_cfg = np.zeros(d, dtype=np.int32)
        c = np.zeros(d, dtype=np.int32)
        rem = np.zeros(d, dtype=np.int32)
        conv = np.zeros(conv_len, dtype=np.float64)
        prefix_c_arr = np.zeros(d + 1, dtype=np.int64)

        c[0] = c0

        # --- d=2 special case ---
        if d == 2:
            c[1] = S - c0
            if c0 <= c[1]:
                a0 = c0 * scale
                a1 = c[1] * scale
                conv[0] = a0 * a0
                conv[1] = conv[0] + 2.0 * a0 * a1
                conv[2] = conv[1] + a1 * a1
                inv_norm = 1.0 / (4.0 * n_half * 2)
                best = 0.0
                for s_lo in range(3):
                    ws = conv[s_lo]
                    if s_lo > 0:
                        ws -= conv[s_lo - 1]
                    tv = ws * inv_norm
                    if tv > best:
                        best = tv
                # Dynamic threshold for ell=2: W_int = c[1]
                dyn_thresh_d2 = c_target + (1.0 + 2.0 * c[1]) * inv_m_sq + fp_margin
                if best > dyn_thresh_d2:
                    local_test += 1
                else:
                    local_surv += 1
                    if best < local_min_tv:
                        local_min_tv = best
                        local_min_cfg[0] = c0
                        local_min_cfg[1] = c[1]
            thread_survivors[c0_idx] = local_surv
            thread_asym[c0_idx] = local_asym
            thread_test[c0_idx] = local_test
            thread_min_tv[c0_idx] = local_min_tv
            for i in range(d):
                thread_min_cfg[c0_idx, i] = local_min_cfg[i]
            continue

        # --- General d >= 3 ---
        rem[1] = S - c0
        pos = 1
        c[1] = 0

        while pos >= 1:
            if pos == d - 2:
                avail = rem[pos]
                c_max = avail - c0

                while c[pos] <= c_max:
                    c[d - 1] = avail - c[pos]

                    is_canon = True
                    for i in range(half_d):
                        j = d - 1 - i
                        if i >= j:
                            break
                        if c[i] < c[j]:
                            break
                        if c[i] > c[j]:
                            is_canon = False
                            break
                    if not is_canon:
                        c[pos] += 1
                        continue

                    max_c = c[0]
                    for i in range(1, d):
                        if c[i] > max_c:
                            max_c = c[i]
                    max_a = max_c * scale
                    if max_a * max_a * inv_ell2 > thresh:
                        local_test += 1
                        c[pos] += 1
                        continue

                    left_sum = 0.0
                    for i in range(half_d):
                        left_sum += c[i]
                    left_frac = left_sum / total_mass
                    if left_frac > 0.5:
                        dom = left_frac
                    else:
                        dom = 1.0 - left_frac
                    asym_base = dom - margin
                    if asym_base < 0.0:
                        asym_base = 0.0
                    asym_val = 2.0 * asym_base * asym_base
                    if asym_val >= c_target:
                        local_asym += 1
                        c[pos] += 1
                        continue

                    for k in range(conv_len):
                        conv[k] = 0.0
                    for ii in range(d):
                        ai = c[ii] * scale
                        conv[2 * ii] += ai * ai
                        for jj in range(ii + 1, d):
                            conv[ii + jj] += 2.0 * ai * (c[jj] * scale)
                    for k in range(1, conv_len):
                        conv[k] += conv[k - 1]

                    # Compute prefix sums of c for dynamic W_int
                    prefix_c_arr[0] = 0
                    for i in range(d):
                        prefix_c_arr[i + 1] = prefix_c_arr[i] + c[i]

                    best = 0.0
                    pruned = False
                    for ell in range(2, d + 1):
                        if pruned:
                            break
                        W_int = prefix_c_arr[ell] - prefix_c_arr[ell // 2]
                        dyn_thresh = c_target + (1.0 + 2.0 * W_int) * inv_m_sq + fp_margin
                        n_cv = ell - 1
                        inv_norm = 1.0 / (4.0 * n_half * ell)
                        for s_lo in range(conv_len - n_cv + 1):
                            s_hi = s_lo + n_cv - 1
                            ws = conv[s_hi]
                            if s_lo > 0:
                                ws -= conv[s_lo - 1]
                            tv = ws * inv_norm
                            if tv > best:
                                best = tv
                            if tv > dyn_thresh:
                                pruned = True
                                break

                    if pruned:
                        local_test += 1
                    else:
                        local_surv += 1
                        if best < local_min_tv:
                            local_min_tv = best
                            for i in range(d):
                                local_min_cfg[i] = c[i]

                    c[pos] += 1

                pos -= 1
                if pos >= 1:
                    c[pos] += 1

            elif c[pos] > rem[pos]:
                pos -= 1
                if pos >= 1:
                    c[pos] += 1

            else:
                prune = False
                next_rem = rem[pos] - c[pos]

                if next_rem < c0:
                    c[pos] = rem[pos] + 1
                    continue

                if not prune:
                    left_partial = 0
                    for i in range(pos + 1):
                        left_partial += c[i]
                    lp = float(left_partial)
                    if lp * lp * norm_d_inv > thresh:
                        prune = True

                if not prune:
                    rr = float(next_rem)
                    if rr * rr * norm_d_inv > thresh:
                        prune = True

                if not prune:
                    max_c = 0
                    for i in range(pos + 1):
                        if c[i] > max_c:
                            max_c = c[i]
                    max_a = max_c * scale
                    if max_a * max_a * inv_ell2 > thresh:
                        prune = True

                if not prune and pos == half_d - 1:
                    left_frac = float(left_partial) / total_mass
                    if left_frac > 0.5:
                        dom = left_frac
                    else:
                        dom = 1.0 - left_frac
                    asym_base = dom - margin
                    if asym_base < 0.0:
                        asym_base = 0.0
                    asym_val = 2.0 * asym_base * asym_base
                    if asym_val >= c_target:
                        prune = True

                if prune:
                    c[pos] += 1
                else:
                    rem[pos + 1] = next_rem
                    pos += 1
                    c[pos] = 0

        thread_survivors[c0_idx] = local_surv
        thread_asym[c0_idx] = local_asym
        thread_test[c0_idx] = local_test
        thread_min_tv[c0_idx] = local_min_tv
        for i in range(d):
            thread_min_cfg[c0_idx, i] = local_min_cfg[i]

    return thread_survivors, thread_asym, thread_test, thread_min_tv, thread_min_cfg


# =====================================================================
# Fused Numba kernels for run_single_level (prove a specific target)
# =====================================================================

@numba.njit(parallel=True, cache=True)
def _prove_target_d4(c0_order, S, n_half, inv_m, margin, prune_target, fp_margin):
    """Fused kernel: prove all d=4 canonical compositions exceed prune_target.

    Returns per-work-unit counts of (survivors, asym_pruned, test_pruned)
    and the minimum test value among survivors.

    M4 optimizations included: ell=4 pair-sum and ell=3 right-window bounds.
    Uses interleaved c0 ordering for balanced parallel load distribution.
    """
    d = 4
    conv_len = 7
    total_mass = float(S)
    n_c0 = len(c0_order)
    thresh = prune_target + fp_margin
    scale = 4.0 * n_half * inv_m   # c -> a conversion factor (4n/m)
    # Asymmetry bound applies directly to c (not through discretization),
    # so compare against c_target, not prune_target.
    c_target = prune_target - 2.0 * inv_m - inv_m * inv_m
    inv_m_sq = inv_m * inv_m

    thread_survivors = np.zeros(n_c0, dtype=np.int64)
    thread_asym = np.zeros(n_c0, dtype=np.int64)
    thread_test = np.zeros(n_c0, dtype=np.int64)
    thread_min_tv = np.full(n_c0, 1e30, dtype=np.float64)
    thread_min_cfg = np.zeros((n_c0, d), dtype=np.int32)

    for c0_idx in numba.prange(n_c0):
        c0 = c0_order[c0_idx]

        r0 = S - c0
        c1_max_canon = r0 - c0
        local_surv = np.int64(0)
        local_asym = np.int64(0)
        local_test = np.int64(0)
        local_min_tv = 1e30
        local_min_cfg = np.zeros(d, dtype=np.int32)
        conv = np.zeros(conv_len, dtype=np.float64)

        for c1 in range(c1_max_canon + 1):
            r1 = r0 - c1
            c2_max = r1 - c0

            # M4: ell=4 pair-sum bound
            norm_ell_d = 4.0 * n_half * d
            pair_left = float(c0 + c1)
            pair_right = float(r1)
            if pair_left * pair_left * scale * scale / norm_ell_d > thresh:
                local_test += c2_max + 1
                continue
            if pair_right * pair_right * scale * scale / norm_ell_d > thresh:
                local_test += c2_max + 1
                continue

            # Asymmetry check
            left_frac = pair_left / total_mass
            if left_frac > 0.5:
                dom = left_frac
            else:
                dom = 1.0 - left_frac
            asym_base = dom - margin
            if asym_base < 0.0:
                asym_base = 0.0
            asym_val = 2.0 * asym_base * asym_base
            if asym_val >= c_target:
                local_asym += c2_max + 1
                continue

            # M4: ell=3 right-window c2 lower bound
            norm_ell3 = 4.0 * n_half * 3
            r1_sq = float(r1) * float(r1)
            ell3_cutoff_sq = r1_sq - thresh * norm_ell3 / (scale * scale)
            if ell3_cutoff_sq > 0.0:
                c2_lo_ell3 = int(math.sqrt(ell3_cutoff_sq)) + 1
            else:
                c2_lo_ell3 = 0
            c2_start = max(0, c2_lo_ell3)

            skipped_ell3 = c2_start
            if skipped_ell3 > 0:
                local_test += min(skipped_ell3, c2_max + 1)
                if c2_start > c2_max:
                    continue

            inv_ell2 = 1.0 / (4.0 * n_half * 2)
            for c2 in range(c2_start, c2_max + 1):
                c3 = r1 - c2
                if c0 == c3 and c1 > c2:
                    continue

                max_c = c3
                if c1 > max_c:
                    max_c = c1
                if c2 > max_c:
                    max_c = c2
                max_a = max_c * scale
                if max_a * max_a * inv_ell2 > thresh:
                    local_test += 1
                    continue

                a0 = c0 * scale
                a1 = c1 * scale
                a2 = c2 * scale
                a3 = c3 * scale
                conv[0] = a0 * a0
                conv[1] = 2.0 * a0 * a1
                conv[2] = a1 * a1 + 2.0 * a0 * a2
                conv[3] = 2.0 * (a0 * a3 + a1 * a2)
                conv[4] = a2 * a2 + 2.0 * a1 * a3
                conv[5] = 2.0 * a2 * a3
                conv[6] = a3 * a3
                for k in range(1, conv_len):
                    conv[k] += conv[k - 1]

                best = 0.0
                pruned = False
                for ell in range(2, d + 1):
                    if pruned:
                        break
                    # Dynamic W_int = sum c[k] for k in [ell//2, ell-1]
                    if ell == 2:
                        W_int = c1
                    elif ell == 3:
                        W_int = c1 + c2
                    else:
                        W_int = c2 + c3
                    dyn_thresh = c_target + (1.0 + 2.0 * W_int) * inv_m_sq + fp_margin
                    n_cv = ell - 1
                    inv_norm = 1.0 / (4.0 * n_half * ell)
                    for s_lo in range(conv_len - n_cv + 1):
                        s_hi = s_lo + n_cv - 1
                        ws = conv[s_hi]
                        if s_lo > 0:
                            ws -= conv[s_lo - 1]
                        tv = ws * inv_norm
                        if tv > best:
                            best = tv
                        if tv > dyn_thresh:
                            pruned = True
                            break

                if pruned:
                    local_test += 1
                else:
                    local_surv += 1
                    if best < local_min_tv:
                        local_min_tv = best
                        local_min_cfg[0] = c0
                        local_min_cfg[1] = c1
                        local_min_cfg[2] = c2
                        local_min_cfg[3] = c3

        thread_survivors[c0_idx] = local_surv
        thread_asym[c0_idx] = local_asym
        thread_test[c0_idx] = local_test
        thread_min_tv[c0_idx] = local_min_tv
        for i in range(d):
            thread_min_cfg[c0_idx, i] = local_min_cfg[i]

    return thread_survivors, thread_asym, thread_test, thread_min_tv, thread_min_cfg


@numba.njit(parallel=True, cache=True)
def _prove_target_d6(c0_order, S, n_half, inv_m, margin, prune_target, fp_margin):
    """Fused kernel: prove all d=6 canonical compositions exceed prune_target.

    M4 optimizations: ell=6 three-bin sum bound at (c0,c1,c2) level.
    Uses interleaved c0 ordering for balanced parallel load distribution.
    """
    d = 6
    conv_len = 11
    total_mass = float(S)
    n_c0 = len(c0_order)
    thresh = prune_target + fp_margin
    scale = 4.0 * n_half * inv_m   # c -> a conversion factor (4n/m)
    # Asymmetry bound applies directly to c (not through discretization),
    # so compare against c_target, not prune_target.
    c_target = prune_target - 2.0 * inv_m - inv_m * inv_m
    inv_m_sq = inv_m * inv_m

    thread_survivors = np.zeros(n_c0, dtype=np.int64)
    thread_asym = np.zeros(n_c0, dtype=np.int64)
    thread_test = np.zeros(n_c0, dtype=np.int64)
    thread_min_tv = np.full(n_c0, 1e30, dtype=np.float64)
    thread_min_cfg = np.zeros((n_c0, d), dtype=np.int32)

    for c0_idx in numba.prange(n_c0):
        c0 = c0_order[c0_idx]
        r0 = S - c0

        local_surv = np.int64(0)
        local_asym = np.int64(0)
        local_test = np.int64(0)
        local_min_tv = 1e30
        local_min_cfg = np.zeros(d, dtype=np.int32)
        conv = np.zeros(conv_len, dtype=np.float64)

        for c1 in range(r0 + 1):
            r1 = r0 - c1
            for c2 in range(r1 + 1):
                r2 = r1 - c2

                # M4: ell=6 three-bin sum
                norm_ell_d = 4.0 * n_half * d
                left3 = float(c0 + c1 + c2)
                right3 = float(r2)
                if left3 * left3 * scale * scale / norm_ell_d > thresh:
                    local_test += 1
                    continue
                if right3 * right3 * scale * scale / norm_ell_d > thresh:
                    local_test += 1
                    continue

                # Asymmetry at (c0,c1,c2) level
                left_frac2 = left3 / total_mass
                if left_frac2 > 0.5:
                    dom2 = left_frac2
                else:
                    dom2 = 1.0 - left_frac2
                asym_base2 = dom2 - margin
                if asym_base2 < 0.0:
                    asym_base2 = 0.0
                asym_val = 2.0 * asym_base2 * asym_base2
                if asym_val >= c_target:
                    local_asym += 1
                    continue

                for c3 in range(r2 + 1):
                    r3 = r2 - c3
                    c4_max = r3 - c0
                    if c4_max < 0:
                        break
                    inv_ell2 = 1.0 / (4.0 * n_half * 2)
                    for c4 in range(c4_max + 1):
                        c5 = r3 - c4
                        if c0 == c5:
                            if c1 > c4:
                                continue
                            if c1 == c4 and c2 > c3:
                                continue

                        max_c = c5
                        if c1 > max_c:
                            max_c = c1
                        if c2 > max_c:
                            max_c = c2
                        if c3 > max_c:
                            max_c = c3
                        if c4 > max_c:
                            max_c = c4
                        max_a_val = max_c * scale
                        if max_a_val * max_a_val * inv_ell2 > thresh:
                            local_test += 1
                            continue

                        a0 = c0 * scale
                        a1 = c1 * scale
                        a2 = c2 * scale
                        a3 = c3 * scale
                        a4 = c4 * scale
                        a5 = c5 * scale
                        conv[0] = a0 * a0
                        conv[1] = 2.0 * a0 * a1
                        conv[2] = 2.0 * a0 * a2 + a1 * a1
                        conv[3] = 2.0 * (a0 * a3 + a1 * a2)
                        conv[4] = 2.0 * (a0 * a4 + a1 * a3) + a2 * a2
                        conv[5] = 2.0 * (a0 * a5 + a1 * a4 + a2 * a3)
                        conv[6] = 2.0 * (a1 * a5 + a2 * a4) + a3 * a3
                        conv[7] = 2.0 * (a2 * a5 + a3 * a4)
                        conv[8] = 2.0 * a3 * a5 + a4 * a4
                        conv[9] = 2.0 * a4 * a5
                        conv[10] = a5 * a5
                        for k in range(1, conv_len):
                            conv[k] += conv[k - 1]

                        best = 0.0
                        pruned = False
                        for ell in range(2, d + 1):
                            if pruned:
                                break
                            # Dynamic W_int = sum c[k] for k in [ell//2, ell-1]
                            if ell == 2:
                                W_int = c1
                            elif ell == 3:
                                W_int = c1 + c2
                            elif ell == 4:
                                W_int = c2 + c3
                            elif ell == 5:
                                W_int = c2 + c3 + c4
                            else:
                                W_int = c3 + c4 + c5
                            dyn_thresh = c_target + (1.0 + 2.0 * W_int) * inv_m_sq + fp_margin
                            n_cv = ell - 1
                            inv_norm = 1.0 / (4.0 * n_half * ell)
                            for s_lo in range(conv_len - n_cv + 1):
                                s_hi = s_lo + n_cv - 1
                                ws = conv[s_hi]
                                if s_lo > 0:
                                    ws -= conv[s_lo - 1]
                                tv = ws * inv_norm
                                if tv > best:
                                    best = tv
                                if tv > dyn_thresh:
                                    pruned = True
                                    break

                        if pruned:
                            local_test += 1
                        else:
                            local_surv += 1
                            if best < local_min_tv:
                                local_min_tv = best
                                local_min_cfg[0] = c0
                                local_min_cfg[1] = c1
                                local_min_cfg[2] = c2
                                local_min_cfg[3] = c3
                                local_min_cfg[4] = c4
                                local_min_cfg[5] = c5

        thread_survivors[c0_idx] = local_surv
        thread_asym[c0_idx] = local_asym
        thread_test[c0_idx] = local_test
        thread_min_tv[c0_idx] = local_min_tv
        for i in range(d):
            thread_min_cfg[c0_idx, i] = local_min_cfg[i]

    return thread_survivors, thread_asym, thread_test, thread_min_tv, thread_min_cfg


# =====================================================================
# Public API
# =====================================================================

def run_single_level(n_half, m, c_target, batch_size=100000, verbose=True):
    """Run the branch-and-prune at a single discretization level.

    For d=4 and d=6, uses fused Numba kernels with M4 window-bound pruning
    (eliminates Python generator overhead). Falls back to batched path for other d.

    Returns
    -------
    dict with keys: proven, c_proven, n_survivors, min_test_val,
                    min_test_config, stats
    """
    d = 2 * n_half
    S = m  # S=m convention: integer coords sum to m (not 4nm)
    n_total = count_compositions(d, S)
    corr = correction(m)
    prune_target = c_target + corr
    asym_thresh = asymmetry_threshold(c_target)
    margin = 1.0 / (4.0 * m)
    inv_m = 1.0 / m
    fp_margin = 1e-9

    if verbose:
        print(f"Level n={n_half}, m={m}: d={d} bins, S={S}")
        print(f"  Grid points: {n_total:,}")
        print(f"  Correction: {corr:.6f}")
        print(f"  Target: C_{{1a}} >= {c_target:.4f}")
        print(f"  Prune threshold: test_val > {prune_target:.6f}")
        print(f"  Asymmetry threshold: left_frac >= {asym_thresh:.4f}")

    t0 = time.time()

    # Fused Numba path: all pruning + test values in one parallel call
    c0_order = _build_interleaved_order(S // 2 + 1)
    if d == 4:
        t_surv, t_asym, t_test, t_min_tv, t_min_cfg = _prove_target_d4(
            c0_order, S, n_half, inv_m, margin, prune_target, fp_margin)
    elif d == 6:
        t_surv, t_asym, t_test, t_min_tv, t_min_cfg = _prove_target_d6(
            c0_order, S, n_half, inv_m, margin, prune_target, fp_margin)
    else:
        t_surv, t_asym, t_test, t_min_tv, t_min_cfg = _prove_target_generic(
            c0_order, S, d, n_half, inv_m, margin, prune_target, fp_margin)

    n_survived = int(t_surv.sum())
    n_pruned_asym = int(t_asym.sum())
    n_pruned_test = int(t_test.sum())
    n_processed = n_pruned_asym + n_pruned_test + n_survived

    if n_survived > 0:
        idx = int(np.argmin(t_min_tv))
        min_test_val = float(t_min_tv[idx])
        min_test_config = t_min_cfg[idx].copy()
    else:
        min_test_val = float('inf')
        min_test_config = None

    survivor_configs = []  # Not tracked in fused path

    elapsed = time.time() - t0
    proven = n_survived == 0
    c_proven = c_target if proven else None

    if verbose:
        print(f"\n  Completed in {elapsed:.1f}s "
              f"({n_processed:,} canonical of {n_total:,} total)")
        print(f"  Asymmetry pruned: {n_pruned_asym:,} "
              f"({100*n_pruned_asym/max(1,n_processed):.1f}%)")
        print(f"  Test pruned: {n_pruned_test:,} "
              f"({100*n_pruned_test/max(1,n_processed):.1f}%)")
        print(f"  Survivors: {n_survived:,} "
              f"({100*n_survived/max(1,n_processed):.1f}%)")
        if proven:
            print(f"  >>> PROVEN: C_{{1a}} >= {c_target:.6f} <<<")
        else:
            print(f"  NOT proven at target {c_target:.4f}")
            if min_test_config is not None:
                a_cfg = min_test_config.astype(np.float64) * (4 * n_half) / m
                print(f"  Min test value: {min_test_val:.6f}")
                print(f"  Min config (a-coords): {a_cfg}")
                print(f"  Min config (mass frac): {a_cfg / (4*n_half)}")

    all_survivors = (np.vstack(survivor_configs) if survivor_configs
                     else np.empty((0, d), dtype=np.int32))

    return {
        'proven': proven,
        'c_proven': c_proven,
        'n_survivors': n_survived,
        'survivors': all_survivors,
        'min_test_val': min_test_val,
        'min_test_config': min_test_config,
        'stats': {
            'n_half': n_half, 'm': m, 'd': d, 'S': S,
            'n_total': n_total, 'n_processed': n_processed,
            'n_pruned_asym': n_pruned_asym,
            'n_pruned_test': n_pruned_test,
            'n_survived': n_survived,
            'elapsed': elapsed,
            'prune_target': prune_target,
            'correction': corr,
            'asym_threshold': asym_thresh,
        },
    }


def find_best_bound(n_half, m, lo=1.0, hi=1.5, tol=0.001,
                     batch_size=100000, verbose=True):
    """Binary search for the best provable bound at given (n, m).

    DEPRECATED: Use find_best_bound_direct() instead for ~8-10x speedup.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Binary search: n={n_half}, m={m}, range=[{lo:.4f}, {hi:.4f}]")
        print(f"{'='*70}")

    best_proven = None

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if verbose:
            print(f"\n--- Trying c_target = {mid:.6f} ---")
        result = run_single_level(n_half, m, mid,
                                   batch_size=batch_size, verbose=verbose)
        if result['proven']:
            lo = mid
            best_proven = mid
        else:
            hi = mid

    if verbose:
        print(f"\n{'='*70}")
        if best_proven is not None:
            print(f"Best proven bound: C_{{1a}} >= {best_proven:.6f}")
        else:
            print(f"Could not prove any bound in [{lo:.4f}, {hi:.4f}]")
        print(f"{'='*70}")

    return best_proven


def find_best_bound_direct(n_half, m, batch_size=50000, verbose=True):
    """Compute the best provable lower bound in a single pass.

    Instead of binary-searching over target values, directly computes:
        bound = min_b max(test_val(b) - correction, asym_bound(b))
    where the min is over all compositions b in B_{n,m}.

    Returns
    -------
    float : best provable lower bound on C_{1a}.
    """
    d = 2 * n_half
    S = m  # S=m convention: integer coords sum to m (not 4nm)
    n_total = count_compositions(d, S)
    corr = correction(m)
    margin = 1.0 / (4.0 * m)
    total_mass = float(S)
    inv_m = 1.0 / m

    # Seed running min from uniform composition
    a_uniform = np.full(d, float(4 * n_half) / d)
    init_tv = compute_test_value_single(a_uniform, n_half)
    min_eff = init_tv - corr
    min_config = None

    if verbose:
        print(f"Single-pass: n={n_half}, m={m}, d={d}, S={S}")
        print(f"  Grid points: {n_total:,}")
        print(f"  Correction: {corr:.6f}")
        print(f"  Initial bound (uniform): {min_eff:.6f}")

    t0 = time.time()

    c0_order = _build_interleaved_order(S // 2 + 1)
    if d == 4:
        thread_mins, thread_cfg = _find_min_eff_d4(
            c0_order, S, n_half, inv_m, margin, corr, min_eff)
    elif d == 6:
        thread_mins, thread_cfg = _find_min_eff_d6(
            c0_order, S, n_half, inv_m, margin, corr, min_eff)
    else:
        thread_mins, thread_cfg = _find_min_eff_generic(
            c0_order, S, d, n_half, inv_m, margin, corr, min_eff)
    idx = int(np.argmin(thread_mins))
    if thread_mins[idx] < min_eff:
        min_eff = float(thread_mins[idx])
        min_config = thread_cfg[idx].copy()

    elapsed = time.time() - t0

    if verbose:
        print(f"  Completed in {elapsed:.1f}s")
        print(f"  >>> PROVEN: C_{{1a}} >= {min_eff:.6f} <<<")
        if min_config is not None:
            a_cfg = min_config.astype(np.float64) * (4 * n_half) / m
            print(f"  Minimizer (a-coords): {a_cfg}")

    return min_eff
