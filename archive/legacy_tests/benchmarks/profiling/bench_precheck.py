"""A/B benchmark: old _tighten_ranges (no pre-check) vs new (with pre-check).

Run on real cascade parents at levels where AC does nothing.
"""
import sys, os, time
import numpy as np
from numba import njit

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "cloninger-steinerberger"))
sys.path.insert(0, os.path.join(_root, "cloninger-steinerberger", "cpu"))
import run_cascade as rc

M, C_TARGET = 20, 1.35


@njit(cache=False)
def _tighten_ranges_old(parent_int, lo_arr, hi_arr, m, c_target, n_half_child,
                         use_flat_threshold=False):
    """Original _tighten_ranges WITHOUT pre-check (builds table immediately)."""
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half_child)
    n_half_d = np.float64(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    S_child_plus_1 = int(4 * n_half_child * m + 1)
    ell_count = conv_len

    # Build full threshold table immediately (no pre-check)
    threshold_table = np.empty(ell_count * S_child_plus_1, dtype=np.int64)
    flat_corr = 2.0 * m_d + 1.0
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        scale_ell = np.float64(ell) * four_n
        if use_flat_threshold:
            dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
            flat_val = np.int64(dyn_x)
            for w in range(S_child_plus_1):
                threshold_table[idx * S_child_plus_1 + w] = flat_val
        else:
            for w in range(S_child_plus_1):
                corr_w = 1.0 + np.float64(w) / (2.0 * n_half_d)
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                threshold_table[idx * S_child_plus_1 + w] = np.int64(dyn_x)

    test_conv = np.empty(conv_len, dtype=np.int64)
    max_rounds = d_parent + 1
    for _round in range(max_rounds):
        any_changed = False
        child_min = np.empty(d_child, dtype=np.int32)
        for q in range(d_parent):
            child_min[2 * q] = lo_arr[q]
            child_min[2 * q + 1] = 2 * parent_int[q] - hi_arr[q]

        conv_min = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            ci = np.int64(child_min[i])
            if ci == 0:
                continue
            conv_min[2 * i] += ci * ci
            for j in range(i + 1, d_child):
                cj = np.int64(child_min[j])
                if cj != 0:
                    conv_min[i + j] += np.int64(2) * ci * cj

        max_child_prefix = np.empty(d_child + 1, dtype=np.int64)
        max_child_prefix[0] = np.int64(0)
        for q in range(d_parent):
            max_child_prefix[2 * q + 1] = (max_child_prefix[2 * q]
                                            + np.int64(hi_arr[q]))
            max_child_prefix[2 * q + 2] = (max_child_prefix[2 * q + 1]
                                            + np.int64(2 * parent_int[q]
                                                       - lo_arr[q]))

        for p in range(d_parent):
            if lo_arr[p] == hi_arr[p]:
                continue
            B_p = parent_int[p]
            k1 = 2 * p
            k2 = 2 * p + 1
            old1 = np.int64(child_min[k1])
            old2 = np.int64(child_min[k2])
            new_lo = lo_arr[p]
            new_hi = hi_arr[p]

            # Tighten from low end
            for v in range(lo_arr[p], hi_arr[p] + 1):
                new1 = np.int64(v)
                new2 = np.int64(2 * B_p - v)
                delta1 = new1 - old1
                delta2 = new2 - old2
                for kk in range(conv_len):
                    test_conv[kk] = conv_min[kk]
                test_conv[2 * k1] += new1 * new1 - old1 * old1
                test_conv[2 * k2] += new2 * new2 - old2 * old2
                test_conv[k1 + k2] += np.int64(2) * (new1 * new2
                                                       - old1 * old2)
                for j in range(d_child):
                    if j == k1 or j == k2:
                        continue
                    cj = np.int64(child_min[j])
                    if cj != 0:
                        test_conv[k1 + j] += np.int64(2) * delta1 * cj
                        test_conv[k2 + j] += np.int64(2) * delta2 * cj
                infeasible = False
                for ell in range(2, 2 * d_child + 1):
                    if infeasible:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    if n_windows <= 0:
                        continue
                    ell_idx = ell - 2
                    ws = np.int64(0)
                    for kk in range(n_cv):
                        ws += test_conv[kk]
                    hb = ell - 2
                    if hb > d_child - 1:
                        hb = d_child - 1
                    W_max = max_child_prefix[hb + 1]
                    if W_max >= np.int64(S_child_plus_1):
                        W_max = np.int64(S_child_plus_1 - 1)
                    if ws > threshold_table[ell_idx * S_child_plus_1
                                            + int(W_max)]:
                        infeasible = True
                        break
                    for s in range(1, n_windows):
                        ws += test_conv[s + n_cv - 1] - test_conv[s - 1]
                        lb = s - (d_child - 1)
                        if lb < 0:
                            lb = 0
                        hb = s + ell - 2
                        if hb > d_child - 1:
                            hb = d_child - 1
                        W_max = (max_child_prefix[hb + 1]
                                 - max_child_prefix[lb])
                        if W_max >= np.int64(S_child_plus_1):
                            W_max = np.int64(S_child_plus_1 - 1)
                        if ws > threshold_table[ell_idx * S_child_plus_1
                                                + int(W_max)]:
                            infeasible = True
                            break
                if infeasible:
                    if v == new_lo:
                        new_lo = v + 1
                    else:
                        break
                else:
                    break

            # Tighten from high end
            for v in range(hi_arr[p], new_lo - 1, -1):
                new1 = np.int64(v)
                new2 = np.int64(2 * B_p - v)
                delta1 = new1 - old1
                delta2 = new2 - old2
                for kk in range(conv_len):
                    test_conv[kk] = conv_min[kk]
                test_conv[2 * k1] += new1 * new1 - old1 * old1
                test_conv[2 * k2] += new2 * new2 - old2 * old2
                test_conv[k1 + k2] += np.int64(2) * (new1 * new2
                                                       - old1 * old2)
                for j in range(d_child):
                    if j == k1 or j == k2:
                        continue
                    cj = np.int64(child_min[j])
                    if cj != 0:
                        test_conv[k1 + j] += np.int64(2) * delta1 * cj
                        test_conv[k2 + j] += np.int64(2) * delta2 * cj
                infeasible = False
                for ell in range(2, 2 * d_child + 1):
                    if infeasible:
                        break
                    n_cv = ell - 1
                    n_windows = conv_len - n_cv + 1
                    if n_windows <= 0:
                        continue
                    ell_idx = ell - 2
                    ws = np.int64(0)
                    for kk in range(n_cv):
                        ws += test_conv[kk]
                    hb = ell - 2
                    if hb > d_child - 1:
                        hb = d_child - 1
                    W_max = max_child_prefix[hb + 1]
                    if W_max >= np.int64(S_child_plus_1):
                        W_max = np.int64(S_child_plus_1 - 1)
                    if ws > threshold_table[ell_idx * S_child_plus_1
                                            + int(W_max)]:
                        infeasible = True
                        break
                    for s in range(1, n_windows):
                        ws += test_conv[s + n_cv - 1] - test_conv[s - 1]
                        lb = s - (d_child - 1)
                        if lb < 0:
                            lb = 0
                        hb = s + ell - 2
                        if hb > d_child - 1:
                            hb = d_child - 1
                        W_max = (max_child_prefix[hb + 1]
                                 - max_child_prefix[lb])
                        if W_max >= np.int64(S_child_plus_1):
                            W_max = np.int64(S_child_plus_1 - 1)
                        if ws > threshold_table[ell_idx * S_child_plus_1
                                                + int(W_max)]:
                            infeasible = True
                            break
                if infeasible:
                    if v == new_hi:
                        new_hi = v - 1
                    else:
                        break
                else:
                    break

            if new_lo != lo_arr[p] or new_hi != hi_arr[p]:
                lo_arr[p] = new_lo
                hi_arr[p] = new_hi
                any_changed = True
                if new_lo > new_hi:
                    return 0
        if not any_changed:
            break

    total = np.int64(1)
    for i in range(d_parent):
        r = hi_arr[i] - lo_arr[i] + 1
        if r <= 0:
            return 0
        total *= np.int64(r)
    return total


def get_cascade_parents():
    """Generate real parents at each level via the d0=2 cascade."""
    l0 = rc.run_level0(n_half=1.0, m=M, c_target=C_TARGET, d0=2,
                        verbose=False)
    parents_l0 = l0['survivors']  # d=2

    # L0->L1
    all_l1 = []
    for i in range(len(parents_l0)):
        surv, _ = rc.process_parent_fused(parents_l0[i], M, C_TARGET, 2)
        if len(surv) > 0:
            all_l1.append(surv)
    parents_l1 = np.concatenate(all_l1)  # d=4

    # L1->L2 (first 200 parents)
    all_l2 = []
    for i in range(min(200, len(parents_l1))):
        surv, _ = rc.process_parent_fused(parents_l1[i], M, C_TARGET, 4)
        if len(surv) > 0:
            all_l2.append(surv)
    parents_l2 = np.concatenate(all_l2) if all_l2 else np.empty(
        (0, 8), dtype=np.int32)  # d=8

    return {
        'L0->L1 (d_child=4)': (parents_l0, 2),
        'L1->L2 (d_child=8)': (parents_l1[:10000], 4),
        'L2->L3 (d_child=16)': (parents_l2, 8),
    }


def bench(func, parents, nhc, label):
    d_ch = 2 * parents.shape[1]
    n = len(parents)
    # Warmup
    result = rc._compute_bin_ranges(parents[0], M, C_TARGET, d_ch, nhc)
    if result is not None:
        lo, hi, _ = result
        func(parents[0], lo, hi, M, C_TARGET, nhc)

    t0 = time.perf_counter()
    for i in range(n):
        result = rc._compute_bin_ranges(parents[i], M, C_TARGET, d_ch, nhc)
        if result is None:
            continue
        lo, hi, _ = result
        func(parents[i], lo, hi, M, C_TARGET, nhc)
    elapsed = time.perf_counter() - t0
    us = elapsed / n * 1e6
    print(f"  {label:8s}: {elapsed:.4f}s  ({us:.1f} us/parent)")
    return elapsed


def verify_identical(parents, nhc):
    """Verify old and new produce identical ranges."""
    d_ch = 2 * parents.shape[1]
    for i in range(len(parents)):
        result = rc._compute_bin_ranges(parents[i], M, C_TARGET, d_ch, nhc)
        if result is None:
            continue
        lo1, hi1, _ = result
        lo2, hi2 = lo1.copy(), hi1.copy()
        lo3, hi3 = lo1.copy(), hi1.copy()
        ta_old = _tighten_ranges_old(parents[i], lo2, hi2, M, C_TARGET, nhc)
        ta_new = rc._tighten_ranges(parents[i], lo3, hi3, M, C_TARGET, nhc)
        if ta_old != ta_new or not np.array_equal(lo2, lo3) or not np.array_equal(hi2, hi3):
            print(f"  MISMATCH at parent {i}: {parents[i]}")
            print(f"    old: lo={lo2} hi={hi2} total={ta_old}")
            print(f"    new: lo={lo3} hi={hi3} total={ta_new}")
            return False
    return True


def main():
    print("Generating real cascade parents (d0=2, m=20, c_target=1.35)...")
    levels = get_cascade_parents()

    for level_name, (parents, nhc) in levels.items():
        if len(parents) == 0:
            continue
        print(f"\n{'='*60}")
        print(f"{level_name}: {len(parents):,} parents")
        print(f"{'='*60}")

        # Verify correctness first
        print("  Verifying identical results...")
        ok = verify_identical(parents, nhc)
        print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            continue

        # Benchmark
        t_old = bench(_tighten_ranges_old, parents, nhc, "OLD")
        t_new = bench(rc._tighten_ranges, parents, nhc, "NEW")
        speedup = t_old / t_new
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
