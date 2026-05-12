"""Test Idea 1: W-refined partial pruning in B&B.

Current code uses flat_thr[ell] for partial pruning at intermediate B&B nodes.
This test implements the tighter W-refined threshold and measures:
  - L0 survivors (baseline vs improved)
  - L1 survivors after expanding L0 survivors
  - L2 survivors after expanding L1 survivors (sample)
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
_l0_bnb_run = _rc._l0_bnb_run
_canonicalize_inplace = _rc._canonicalize_inplace
_fast_dedup = _rc._fast_dedup
process_parent_fused = _rc.process_parent_fused
run_level0 = _rc.run_level0


# =====================================================================
# Modified B&B inner kernel with W-refined partial pruning
# =====================================================================

@njit(cache=True)
def _l0_bnb_inner_wref(c0, d, S, n_half_d, cs_base_m2, eps_margin,
                        flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                        max_ell, four_n, use_flat_threshold,
                        out_buf, count_only):
    """B&B subtree with W-refined partial pruning.

    Key difference from baseline: at intermediate nodes, compute partial
    W_int from assigned bins and use the W-refined threshold instead of
    the flat threshold.  This is sound because:
      - partial_ws <= final_ws  (all cross-terms are non-negative)
      - partial_W_int <= final_W_int  (additional bins can only add mass)
      - therefore partial_W_refined_thr <= final_W_refined_thr
    """
    conv_len = 2 * d - 1
    d_m1 = d - 1
    conv = np.zeros(conv_len, dtype=np.int32)
    bins = np.zeros(d, dtype=np.int32)
    rem_arr = np.zeros(d, dtype=np.int32)

    bins[0] = c0
    if c0 > 0:
        conv[0] = c0 * c0
    rem_arr[0] = S
    rem_arr[1] = S - c0

    n_surv = np.int64(0)
    n_tested = np.int64(0)
    buf_cap = np.int64(0)
    if not count_only:
        buf_cap = np.int64(out_buf.shape[0])

    if d == 1:
        if c0 == S:
            n_tested = 1
        return n_surv, n_tested

    if d == 2:
        forced = S - c0
        if 0 <= forced <= x_cap:
            n_tested = 1
            bins[1] = forced
            conv[0] = c0 * c0
            conv[1] = np.int32(2) * c0 * np.int32(forced)
            conv[2] = forced * forced

            left_s = np.float64(bins[0])
            left_frac = left_s / np.float64(S)
            asym_cov = (left_frac <= 1.0 - asym_thr) or (left_frac >= asym_thr)

            if not asym_cov:
                pruned_leaf = False
                if use_flat_threshold:
                    for ell in range(2, max_ell + 1):
                        if pruned_leaf:
                            break
                        n_cv = ell - 1
                        nw = conv_len - n_cv + 1
                        ws = np.int64(0)
                        for k in range(n_cv):
                            ws += np.int64(conv[k])
                        for s_lo in range(nw):
                            if s_lo > 0:
                                ws += (np.int64(conv[s_lo + n_cv - 1])
                                       - np.int64(conv[s_lo - 1]))
                            if ws > flat_thr[ell]:
                                pruned_leaf = True
                                break
                else:
                    prefix_c = np.zeros(d + 1, dtype=np.int64)
                    for i in range(d):
                        prefix_c[i + 1] = prefix_c[i] + np.int64(bins[i])
                    for ell in range(2, max_ell + 1):
                        if pruned_leaf:
                            break
                        n_cv = ell - 1
                        nw = conv_len - n_cv + 1
                        ws = np.int64(0)
                        for k in range(n_cv):
                            ws += np.int64(conv[k])
                        sc = scale_arr[ell]
                        for s_lo in range(nw):
                            if s_lo > 0:
                                ws += (np.int64(conv[s_lo + n_cv - 1])
                                       - np.int64(conv[s_lo - 1]))
                            lo_b = s_lo - d_m1
                            if lo_b < 0:
                                lo_b = 0
                            hi_b = s_lo + ell - 2
                            if hi_b > d_m1:
                                hi_b = d_m1
                            W_int = prefix_c[hi_b + 1] - prefix_c[lo_b]
                            cw = 1.0 + np.float64(W_int) / (2.0 * n_half_d)
                            dyn_it = np.int64(
                                (cs_base_m2 + cw + eps_margin) * sc)
                            if ws > dyn_it:
                                pruned_leaf = True
                                break
                if not pruned_leaf:
                    if not count_only and n_surv < buf_cap:
                        out_buf[n_surv, 0] = c0
                        out_buf[n_surv, 1] = forced
                    n_surv += 1
        return n_surv, n_tested

    # General case: d >= 3
    pos = 1
    bins[1] = 0

    # Prefix sum of assigned bins for W_int computation
    prefix_partial = np.zeros(d + 1, dtype=np.int64)
    prefix_partial[1] = np.int64(c0)

    while True:
        c_val = bins[pos]
        rem = rem_arr[pos]

        if pos == d_m1:
            # --- Last bin: forced value ---
            forced = rem
            if 0 <= forced <= x_cap:
                n_tested += 1
                bins[pos] = forced

                conv[2 * pos] += forced * forced
                for j in range(pos):
                    conv[pos + j] += np.int32(2) * forced * bins[j]

                left_s = np.float64(0)
                for i in range(left_bins):
                    left_s += np.float64(bins[i])
                left_frac = left_s / np.float64(S)
                asym_cov = ((left_frac <= 1.0 - asym_thr)
                            or (left_frac >= asym_thr))

                if not asym_cov:
                    pruned_leaf = False
                    if use_flat_threshold:
                        for ell in range(2, max_ell + 1):
                            if pruned_leaf:
                                break
                            n_cv = ell - 1
                            nw = conv_len - n_cv + 1
                            ws = np.int64(0)
                            for k in range(n_cv):
                                ws += np.int64(conv[k])
                            thr_val = flat_thr[ell]
                            for s_lo in range(nw):
                                if s_lo > 0:
                                    ws += (np.int64(conv[s_lo + n_cv - 1])
                                           - np.int64(conv[s_lo - 1]))
                                if ws > thr_val:
                                    pruned_leaf = True
                                    break
                    else:
                        prefix_c = np.zeros(d + 1, dtype=np.int64)
                        for i in range(d):
                            prefix_c[i + 1] = (prefix_c[i]
                                               + np.int64(bins[i]))
                        for ell in range(2, max_ell + 1):
                            if pruned_leaf:
                                break
                            n_cv = ell - 1
                            nw = conv_len - n_cv + 1
                            ws = np.int64(0)
                            for k in range(n_cv):
                                ws += np.int64(conv[k])
                            sc = scale_arr[ell]
                            for s_lo in range(nw):
                                if s_lo > 0:
                                    ws += (
                                        np.int64(conv[s_lo + n_cv - 1])
                                        - np.int64(conv[s_lo - 1]))
                                lo_b = s_lo - d_m1
                                if lo_b < 0:
                                    lo_b = 0
                                hi_b = s_lo + ell - 2
                                if hi_b > d_m1:
                                    hi_b = d_m1
                                W_int = (prefix_c[hi_b + 1]
                                         - prefix_c[lo_b])
                                cw = (1.0 + np.float64(W_int)
                                      / (2.0 * n_half_d))
                                dyn_it = np.int64(
                                    (cs_base_m2 + cw + eps_margin) * sc)
                                if ws > dyn_it:
                                    pruned_leaf = True
                                    break

                    if not pruned_leaf:
                        if not count_only and n_surv < buf_cap:
                            for i in range(d):
                                out_buf[n_surv, i] = bins[i]
                        n_surv += 1

                conv[2 * pos] -= forced * forced
                for j in range(pos):
                    conv[pos + j] -= np.int32(2) * forced * bins[j]

            # Backtrack
            pos -= 1
            if pos < 1:
                break
            c_old = bins[pos]
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int32(2) * c_old * bins[j]
            bins[pos] += 1
            continue

        # --- Non-last bin ---
        max_v = min(rem, x_cap)
        min_v = rem - (d_m1 - pos) * x_cap
        if min_v < 0:
            min_v = 0
        if c_val < min_v:
            bins[pos] = min_v
            c_val = min_v

        if c_val > max_v:
            if pos <= 1:
                break
            pos -= 1
            c_old = bins[pos]
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int32(2) * c_old * bins[j]
            bins[pos] += 1
            continue

        # Add conv contribution
        if c_val > 0:
            conv[2 * pos] += c_val * c_val
            for j in range(pos):
                conv[pos + j] += np.int32(2) * c_val * bins[j]

        # Update partial prefix sum
        prefix_partial[pos + 1] = prefix_partial[pos] + np.int64(c_val)

        # === W-REFINED PARTIAL PRUNE (the improvement) ===
        pruned_partial = False
        max_ci = 2 * pos
        for ell in range(2, max_ell + 1):
            if pruned_partial:
                break
            n_cv = ell - 1
            max_s = min(max_ci, conv_len - n_cv)
            if max_s < 0:
                continue
            ws = np.int64(0)
            init_end = n_cv
            if init_end > max_ci + 1:
                init_end = max_ci + 1
            for k in range(init_end):
                ws += np.int64(conv[k])

            if use_flat_threshold:
                thr_val = flat_thr[ell]
                if ws > thr_val:
                    pruned_partial = True
                    break
                for s_lo in range(1, max_s + 1):
                    new_k = s_lo + n_cv - 1
                    if new_k <= max_ci:
                        ws += np.int64(conv[new_k])
                    ws -= np.int64(conv[s_lo - 1])
                    if ws > thr_val:
                        pruned_partial = True
                        break
            else:
                # W-refined: compute partial W_int from assigned bins
                sc = scale_arr[ell]
                # For s_lo=0: window covers conv indices [0, n_cv-1]
                # Bins overlapping: lo_b..hi_b
                lo_b = 0
                hi_b = ell - 2
                if hi_b > pos:  # only assigned bins up to pos
                    hi_b = pos
                W_int_partial = prefix_partial[hi_b + 1] - prefix_partial[lo_b]
                cw = 1.0 + np.float64(W_int_partial) / (2.0 * n_half_d)
                thr_val = np.int64((cs_base_m2 + cw + eps_margin) * sc)
                if ws > thr_val:
                    pruned_partial = True
                    break
                for s_lo in range(1, max_s + 1):
                    new_k = s_lo + n_cv - 1
                    if new_k <= max_ci:
                        ws += np.int64(conv[new_k])
                    ws -= np.int64(conv[s_lo - 1])
                    # Recompute W_int for this window position
                    lo_b = s_lo - d_m1
                    if lo_b < 0:
                        lo_b = 0
                    hi_b = s_lo + ell - 2
                    if hi_b > pos:  # clamp to assigned range
                        hi_b = pos
                    if hi_b < lo_b:
                        # Window doesn't overlap any assigned bin
                        W_int_partial = np.int64(0)
                    else:
                        W_int_partial = (prefix_partial[hi_b + 1]
                                         - prefix_partial[lo_b])
                    cw = 1.0 + np.float64(W_int_partial) / (2.0 * n_half_d)
                    thr_val = np.int64((cs_base_m2 + cw + eps_margin) * sc)
                    if ws > thr_val:
                        pruned_partial = True
                        break

        if pruned_partial:
            if c_val > 0:
                conv[2 * pos] -= c_val * c_val
                for j in range(pos):
                    conv[pos + j] -= np.int32(2) * c_val * bins[j]
            bins[pos] += 1
            continue

        # Descend
        rem_arr[pos + 1] = rem - c_val
        pos += 1
        bins[pos] = 0

    return n_surv, n_tested


@njit(parallel=True, cache=True)
def _l0_bnb_count_wref(d, S, n_half_d, cs_base_m2, eps_margin,
                        flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                        max_ell, four_n, use_flat_threshold,
                        min_c0, n_c0, counts, tested_arr):
    dummy = np.empty((0, d), dtype=np.int32)
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        ns, nt = _l0_bnb_inner_wref(
            c0, d, S, n_half_d, cs_base_m2, eps_margin,
            flat_thr, scale_arr, x_cap, asym_thr, left_bins,
            max_ell, four_n, use_flat_threshold, dummy, True)
        counts[idx] = ns
        tested_arr[idx] = nt


@njit(parallel=True, cache=True)
def _l0_bnb_fill_wref(d, S, n_half_d, cs_base_m2, eps_margin,
                       flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                       max_ell, four_n, use_flat_threshold,
                       min_c0, n_c0, counts, offsets, out_buf):
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        cnt = counts[idx]
        if cnt == 0:
            continue
        off = offsets[idx]
        _l0_bnb_inner_wref(
            c0, d, S, n_half_d, cs_base_m2, eps_margin,
            flat_thr, scale_arr, x_cap, asym_thr, left_bins,
            max_ell, four_n, use_flat_threshold,
            out_buf[off:off + cnt], False)


def l0_bnb_run_wref(d, S, n_half, m, c_target, use_flat_threshold):
    """Run parallel B&B L0 with W-refined partial pruning."""
    m_d = np.float64(m)
    n_half_d = np.float64(n_half)
    four_n = 4.0 * n_half_d
    cs_base_m2 = c_target * m_d * m_d
    eps_margin = 1e-9 * m_d * m_d
    max_ell = 2 * d
    d_m1 = d - 1

    flat_corr = 2.0 * m_d + 1.0
    flat_thr = np.empty(max_ell + 1, dtype=np.int64)
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n
        flat_thr[ell] = np.int64(
            (cs_base_m2 + flat_corr + eps_margin) * scale_arr[ell])

    thresh_xcap = c_target + 2.0 / m_d + 1.0 / (m_d * m_d) + 1e-9
    x_cap = np.int32(np.floor(
        m_d * np.sqrt(4.0 * np.float64(d) * thresh_xcap)))
    asym_thr = np.sqrt(c_target / 2.0)
    left_bins = d // 2

    min_c0 = max(0, S - d_m1 * int(x_cap))
    max_c0 = min(S, int(x_cap))
    max_c0_flat = int(np.floor(np.sqrt(float(flat_thr[2]))))
    if max_c0_flat < max_c0:
        max_c0 = max_c0_flat

    n_c0 = max_c0 - min_c0 + 1
    if n_c0 <= 0:
        return np.empty((0, d), dtype=np.int32), 0, 0

    counts = np.zeros(n_c0, dtype=np.int64)
    tested_arr = np.zeros(n_c0, dtype=np.int64)

    _l0_bnb_count_wref(d, S, n_half_d, cs_base_m2, eps_margin,
                        flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                        max_ell, four_n, use_flat_threshold,
                        min_c0, n_c0, counts, tested_arr)

    offsets = np.zeros(n_c0 + 1, dtype=np.int64)
    for i in range(n_c0):
        offsets[i + 1] = offsets[i] + counts[i]
    total_surv = int(offsets[n_c0])
    total_tested = int(np.sum(tested_arr))

    if total_surv == 0:
        return np.empty((0, d), dtype=np.int32), 0, total_tested

    out_buf = np.empty((total_surv, d), dtype=np.int32)

    _l0_bnb_fill_wref(d, S, n_half_d, cs_base_m2, eps_margin,
                       flat_thr, scale_arr, x_cap, asym_thr, left_bins,
                       max_ell, four_n, use_flat_threshold,
                       min_c0, n_c0, counts, offsets, out_buf)

    return out_buf, total_surv, total_tested


# =====================================================================
# Comparison harness
# =====================================================================

def run_comparison(d0, m, c_target, max_l1_parents=50, max_l2_parents=20):
    """Run baseline vs W-refined B&B and compare L0, L1, L2 survivors."""
    d = d0
    n_half = d / 2.0
    S = int(2 * d * m)
    use_flat = False  # W-refined leaf pruning (production setting)

    print(f"{'='*70}")
    print(f"Config: d0={d0}, m={m}, c_target={c_target}, S={S}")
    print(f"{'='*70}")

    # --- L0: Baseline ---
    t0 = time.time()
    surv_base, n_surv_base, n_tested_base = _l0_bnb_run(
        d, S, n_half, m, c_target, use_flat)
    if len(surv_base) > 0:
        mask = _canonical_mask(surv_base)
        surv_base = surv_base[mask]
    t_base = time.time() - t0

    # --- L0: W-refined partial ---
    t0 = time.time()
    surv_wref, n_surv_wref, n_tested_wref = l0_bnb_run_wref(
        d, S, n_half, m, c_target, use_flat)
    if len(surv_wref) > 0:
        mask = _canonical_mask(surv_wref)
        surv_wref = surv_wref[mask]
    t_wref = time.time() - t0

    print(f"\n--- L0 (d={d}) ---")
    print(f"  Baseline:  {len(surv_base):,} survivors, "
          f"{n_tested_base:,} leaves tested, {t_base:.3f}s")
    print(f"  W-refined: {len(surv_wref):,} survivors, "
          f"{n_tested_wref:,} leaves tested, {t_wref:.3f}s")
    print(f"  Leaves saved: {n_tested_base - n_tested_wref:,} "
          f"({100*(n_tested_base - n_tested_wref)/max(n_tested_base,1):.1f}%)")
    print(f"  Survivor difference: {len(surv_base) - len(surv_wref)}")

    # Verify W-refined is a subset of baseline
    if len(surv_wref) > 0 and len(surv_base) > 0:
        set_base = set(tuple(r) for r in surv_base)
        set_wref = set(tuple(r) for r in surv_wref)
        if not set_wref.issubset(set_base):
            print("  WARNING: W-refined has survivors NOT in baseline!")
            extra = set_wref - set_base
            print(f"  Extra: {list(extra)[:5]}")
        else:
            print("  VERIFIED: W-refined survivors are subset of baseline")

    # --- L1: expand L0 survivors ---
    n_half_child = d  # d_child/2 = 2*d/2 = d
    d_child = 2 * d

    def expand_to_l1(l0_survivors, label, max_parents):
        n_parents = min(len(l0_survivors), max_parents)
        total_children = 0
        total_l1_surv = 0
        t0 = time.time()
        for i in range(n_parents):
            parent = l0_survivors[i]
            surv, tc = process_parent_fused(parent, m, c_target, n_half_child)
            total_children += tc
            total_l1_surv += len(surv)
        elapsed = time.time() - t0
        print(f"  {label}: {n_parents} parents -> {total_children:,} children "
              f"-> {total_l1_surv:,} L1 survivors ({elapsed:.2f}s)")
        return total_l1_surv, total_children

    print(f"\n--- L0->L1 (d={d} -> d={d_child}) ---")
    l1_surv_base, l1_ch_base = expand_to_l1(surv_base, "Baseline ", max_l1_parents)
    l1_surv_wref, l1_ch_wref = expand_to_l1(surv_wref, "W-refined", max_l1_parents)
    if l1_ch_base > 0:
        print(f"  L1 children saved: {l1_ch_base - l1_ch_wref:,}")
        print(f"  L1 survivors saved: {l1_surv_base - l1_surv_wref:,}")

    # --- L2: expand sample of L1 survivors ---
    if max_l2_parents > 0 and len(surv_base) > 0:
        n_half_l2 = 2 * d  # d_child_l2/2
        d_child_l2 = 4 * d

        def expand_to_l2(l0_survivors, label, max_l1, max_l2):
            # Get L1 survivors from first few L0 parents
            l1_all = []
            for i in range(min(len(l0_survivors), max_l1)):
                surv, _ = process_parent_fused(
                    l0_survivors[i], m, c_target, d)  # n_half_child = d
                if len(surv) > 0:
                    l1_all.append(surv)
            if not l1_all:
                print(f"  {label}: no L1 survivors to expand")
                return 0, 0
            l1_all = np.vstack(l1_all)

            n_parents = min(len(l1_all), max_l2)
            total_children = 0
            total_l2_surv = 0
            t0 = time.time()
            for i in range(n_parents):
                parent = l1_all[i]
                surv, tc = process_parent_fused(
                    parent, m, c_target, 2 * d)  # n_half_child = 2*d
                total_children += tc
                total_l2_surv += len(surv)
            elapsed = time.time() - t0
            print(f"  {label}: {n_parents} L1 parents -> {total_children:,} "
                  f"children -> {total_l2_surv:,} L2 survivors ({elapsed:.2f}s)")
            return total_l2_surv, total_children

        print(f"\n--- L1->L2 (d={d_child} -> d={d_child_l2}) ---")
        l2_s_b, l2_c_b = expand_to_l2(surv_base, "Baseline ", max_l1_parents, max_l2_parents)
        l2_s_w, l2_c_w = expand_to_l2(surv_wref, "W-refined", max_l1_parents, max_l2_parents)
        if l2_c_b > 0:
            print(f"  L2 children saved: {l2_c_b - l2_c_w:,}")
            print(f"  L2 survivors saved: {l2_s_b - l2_s_w:,}")


if __name__ == '__main__':
    print("Warming up JIT...")
    # Warmup
    d_w, S_w = 2, 28
    _l0_bnb_run(d_w, S_w, 1.0, 7, 1.28, False)
    l0_bnb_run_wref(d_w, S_w, 1.0, 7, 1.28, False)
    print("JIT warm, starting tests.\n")

    # Test at multiple configs
    configs = [
        # (d0, m, c_target, max_l1_parents, max_l2_parents)
        (2, 7, 1.28, 50, 20),
        (2, 20, 1.28, 50, 10),
        (2, 7, 1.40, 50, 20),
        (2, 20, 1.40, 50, 10),
        (3, 7, 1.28, 30, 10),
        (4, 7, 1.28, 20, 5),
    ]

    for d0, m, c_target, ml1, ml2 in configs:
        run_comparison(d0, m, c_target, ml1, ml2)
        print()
