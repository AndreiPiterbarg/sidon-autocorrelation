"""Benchmark the four pruning proposals on levels where pruning actually fires.

Uses n_half=3, m=15, c_target=1.33 cascade where:
  L1: 71.6% prune rate (d_parent=6, d_child=12)
  L2: 99.4% prune rate (d_parent=12, d_child=24)
  L3: ~100% prune rate (d_parent=24, d_child=48)

Measures:
  - Proposal 1: children tested with multi-level subtree vs. j=7 only
  - Proposal 2: extra subtree prunes from min unfixed contributions
  - Proposal 3: Cartesian product reduction from arc consistency (W_int_max)
  - Proposal 4: extra subtree prunes from partial-overlap windows
  - Combined: all proposals together vs. reference

Usage:
    python tests/bench_pruning_proposals.py
    python tests/bench_pruning_proposals.py --level 2
"""

import argparse
import math
import os
import sys
import time
import itertools

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_repo_dir, "cloninger-steinerberger")
sys.path.insert(0, _cs_dir)


def _import_cascade():
    cascade_path = os.path.join(_cs_dir, "cpu", "run_cascade.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_cascade", cascade_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cascade = _import_cascade()
_compute_bin_ranges = _cascade._compute_bin_ranges
process_parent_fused = _cascade.process_parent_fused


def compute_autoconv(child):
    d = len(child)
    conv = np.zeros(2 * d - 1, dtype=np.int64)
    for i in range(d):
        ci = int(child[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(child[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj
    return conv


def compute_threshold_table(m, c_target, d_child, n_half_child):
    inv_4n = 1.0 / (4.0 * float(n_half_child))
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * float(m) * float(m)
    c_target_m2 = c_target * float(m) * float(m)
    ell_count = 2 * d_child - 1
    m_plus_1 = m + 1
    table = np.empty(ell_count * m_plus_1, dtype=np.int64)
    for ell in range(2, 2 * d_child + 1):
        idx = ell - 2
        cs_base = (c_target_m2 + 3.0 + eps_margin) * float(ell) * inv_4n
        w_scale = 2.0 * float(ell) * inv_4n
        for w in range(m_plus_1):
            dyn_x = cs_base + w_scale * float(w)
            table[idx * m_plus_1 + w] = int(dyn_x * one_minus_4eps)
    return table


def _subtree_check(child, fixed_len, d_parent, d_child, m, threshold_table,
                   parent_prefix, lo_arr, hi_arr, use_min_unfixed=False,
                   use_partial_overlap=False):
    """Check if a subtree can be pruned.

    Parameters
    ----------
    use_min_unfixed : bool (Proposal 2)
    use_partial_overlap : bool (Proposal 4)
    """
    conv_len = 2 * d_child - 1
    m_plus_1 = m + 1

    partial_conv_fixed = compute_autoconv(child[:fixed_len])
    partial_conv_len = 2 * fixed_len - 1

    prefix_c = np.zeros(fixed_len + 1, dtype=np.int64)
    for i in range(fixed_len):
        prefix_c[i + 1] = prefix_c[i] + int(child[i])

    fixed_parent_boundary = fixed_len // 2

    # Build enhanced conv if using proposals 2 or 4
    if use_min_unfixed or use_partial_overlap:
        enhanced = np.zeros(conv_len, dtype=np.int64)
        for k in range(partial_conv_len):
            enhanced[k] += int(partial_conv_fixed[k])

        # Add minimum contributions from unfixed bins
        for p in range(fixed_parent_boundary, d_parent):
            min_lo = int(lo_arr[p])
            min_hi = int(hi_arr[p])  # This is parent[p] - hi_arr[p]... no
            # Actually: child[2p] >= lo_arr[p], child[2p+1] = parent[p] - child[2p]
            # child[2p+1] >= parent[p] - hi_arr[p]
            min_2p = int(lo_arr[p])
            min_2p1 = int(parent_prefix[p + 1] - parent_prefix[p]) - int(hi_arr[p])
            k_lo = 2 * p
            k_hi = 2 * p + 1

            if 2 * k_lo < conv_len:
                enhanced[2 * k_lo] += min_2p * min_2p
            if 2 * k_hi < conv_len:
                enhanced[2 * k_hi] += min_2p1 * min_2p1
            if k_lo + k_hi < conv_len:
                enhanced[k_lo + k_hi] += 2 * min_2p * min_2p1

            for q in range(fixed_len):
                cq = int(child[q])
                if cq != 0:
                    if q + k_lo < conv_len:
                        enhanced[q + k_lo] += 2 * cq * min_2p
                    if q + k_hi < conv_len:
                        enhanced[q + k_hi] += 2 * cq * min_2p1

        # Make prefix sum
        enhanced_ps = enhanced.copy()
        for k in range(1, conv_len):
            enhanced_ps[k] += enhanced_ps[k - 1]

    # Make prefix sum of partial conv (fixed-only)
    pcs = np.zeros(partial_conv_len, dtype=np.int64)
    for k in range(partial_conv_len):
        pcs[k] = int(partial_conv_fixed[k])
    for k in range(1, partial_conv_len):
        pcs[k] += pcs[k - 1]

    for ell in range(2, 2 * d_child + 1):
        ell_idx = ell - 2
        n_cv = ell - 1

        # Determine window range
        if use_partial_overlap:
            max_windows = conv_len - n_cv + 1
        else:
            max_windows = partial_conv_len - n_cv + 1

        if max_windows <= 0:
            continue

        for s_lo in range(max_windows):
            s_hi = s_lo + n_cv - 1

            # Window sum
            if use_min_unfixed or use_partial_overlap:
                ws = int(enhanced_ps[s_hi])
                if s_lo > 0:
                    ws -= int(enhanced_ps[s_lo - 1])
            else:
                if s_hi >= partial_conv_len:
                    continue
                ws = int(pcs[s_hi])
                if s_lo > 0:
                    ws -= int(pcs[s_lo - 1])

            # W_int_max
            lo_bin = max(0, s_lo - (d_child - 1))
            hi_bin = min(d_child - 1, s_lo + ell - 2)

            fixed_hi = min(hi_bin, fixed_len - 1)
            W_fixed = 0
            if fixed_hi >= lo_bin:
                W_fixed = sum(int(child[i]) for i in range(max(0, lo_bin), fixed_hi + 1))

            unfixed_lo = max(lo_bin, fixed_len)
            W_unfixed = 0
            if unfixed_lo <= hi_bin:
                p_lo = max(unfixed_lo // 2, fixed_parent_boundary)
                p_hi = min(hi_bin // 2, d_parent - 1)
                if p_lo <= p_hi:
                    W_unfixed = int(parent_prefix[p_hi + 1] - parent_prefix[p_lo])

            W_max = min(W_fixed + W_unfixed, m)
            dyn_it = threshold_table[ell_idx * m_plus_1 + W_max]

            if ws > dyn_it:
                return True

    return False


def arc_consistency_tighten(parent_int, m, c_target, d_child, n_half_child,
                             lo_arr, hi_arr):
    """Proposal 3: tighten cursor ranges using W_int_max (corrected)."""
    d_parent = len(parent_int)
    threshold_table = compute_threshold_table(m, c_target, d_child, n_half_child)
    conv_len = 2 * d_child - 1
    m_plus_1 = m + 1

    lo_t = lo_arr.copy()
    hi_t = hi_arr.copy()
    changed = True
    n_rounds = 0

    while changed and n_rounds < 10:
        changed = False
        n_rounds += 1

        for p in range(d_parent):
            B_p = int(parent_int[p])
            new_lo = lo_t[p]
            new_hi = hi_t[p]

            for v in range(lo_t[p], hi_t[p] + 1):
                v1, v2 = v, B_p - v
                infeasible = False

                for ell in range(2, 2 * d_child + 1):
                    if infeasible:
                        break
                    ell_idx = ell - 2
                    n_cv = ell - 1

                    for s_lo in range(conv_len - n_cv + 1):
                        s_hi_cv = s_lo + n_cv - 1

                        self_ws = 0
                        if s_lo <= 4 * p <= s_hi_cv:
                            self_ws += v1 * v1
                        if s_lo <= 4 * p + 2 <= s_hi_cv:
                            self_ws += v2 * v2
                        if s_lo <= 4 * p + 1 <= s_hi_cv:
                            self_ws += 2 * v1 * v2

                        min_other = 0
                        for q in range(d_parent):
                            if q == p:
                                continue
                            mq1 = int(lo_t[q])
                            mq2 = int(parent_int[q]) - int(hi_t[q])

                            if s_lo <= 4 * q <= s_hi_cv:
                                min_other += mq1 * mq1
                            if s_lo <= 4 * q + 2 <= s_hi_cv:
                                min_other += mq2 * mq2
                            if s_lo <= 4 * q + 1 <= s_hi_cv:
                                min_other += 2 * mq1 * mq2

                            if s_lo <= 2*p + 2*q <= s_hi_cv:
                                min_other += 2 * v1 * mq1
                            if s_lo <= 2*p + 2*q+1 <= s_hi_cv:
                                min_other += 2 * v1 * mq2
                            if s_lo <= 2*p+1 + 2*q <= s_hi_cv:
                                min_other += 2 * v2 * mq1
                            if s_lo <= 2*p+1 + 2*q+1 <= s_hi_cv:
                                min_other += 2 * v2 * mq2

                        total = self_ws + min_other

                        lo_bin = max(0, s_lo - (d_child - 1))
                        hi_bin = min(d_child - 1, s_lo + ell - 2)
                        W_max = 0
                        for i in range(lo_bin, hi_bin + 1):
                            pp = i // 2
                            if pp == p:
                                W_max += v1 if i % 2 == 0 else v2
                            else:
                                if i % 2 == 0:
                                    W_max += int(hi_t[pp])
                                else:
                                    W_max += int(parent_int[pp]) - int(lo_t[pp])
                        W_max = min(W_max, m)

                        if total > threshold_table[ell_idx * m_plus_1 + W_max]:
                            infeasible = True
                            break
                    if infeasible:
                        break

                if infeasible:
                    if v == new_lo:
                        new_lo = v + 1
                    elif v == new_hi:
                        new_hi = v - 1

            if new_lo != lo_t[p] or new_hi != hi_t[p]:
                lo_t[p] = new_lo
                hi_t[p] = new_hi
                changed = True

    return lo_t, hi_t


def bench_proposals(parents, m, c_target, n_half_child, max_parents=50,
                     level_name=""):
    """Benchmark all proposals on given parents."""
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_parents = min(len(parents), max_parents)

    threshold_table = compute_threshold_table(m, c_target, d_child, n_half_child)

    # Counters
    total_tc_orig = 0
    total_tc_p3 = 0        # After arc consistency
    total_surv_ref = 0
    total_subtree_checks = 0
    subtree_orig = 0       # j=7 only, fixed-only windows
    subtree_p1 = 0         # multi-level (j>=2)
    subtree_p2 = 0         # multi-level + min unfixed
    subtree_p4 = 0         # multi-level + min unfixed + partial overlap
    total_children_ref = 0
    total_children_p1_skip = 0

    t_start = time.time()

    for pidx in range(n_parents):
        parent = parents[pidx]

        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            continue
        lo_arr, hi_arr, tc_orig = result
        total_tc_orig += tc_orig

        # Reference: run kernel
        surv, tc = process_parent_fused(parent, m, c_target, n_half_child)
        total_surv_ref += len(surv)
        total_children_ref += tc

        # Proposal 3: arc consistency
        lo_t, hi_t = arc_consistency_tighten(
            parent, m, c_target, d_child, n_half_child, lo_arr, hi_arr)
        tc_p3 = 1
        for i in range(d_parent):
            r = hi_t[i] - lo_t[i] + 1
            if r <= 0:
                tc_p3 = 0
                break
            tc_p3 *= r
        total_tc_p3 += tc_p3

        # Proposals 1/2/4: simulate Gray code and count subtree prune opportunities
        parent_prefix = np.zeros(d_parent + 1, dtype=np.int64)
        for i in range(d_parent):
            parent_prefix[i + 1] = parent_prefix[i] + int(parent[i])

        n_active = 0
        active_pos = []
        radix_arr = []
        for i in range(d_parent - 1, -1, -1):
            r = hi_arr[i] - lo_arr[i] + 1
            if r > 1:
                active_pos.append(i)
                radix_arr.append(r)
                n_active += 1

        if n_active < 3:
            continue

        # Run Gray code, checking subtree prune at each j advance
        cursor = np.array([lo_arr[i] for i in range(d_parent)], dtype=np.int32)
        child = np.empty(d_child, dtype=np.int32)
        for i in range(d_parent):
            child[2 * i] = cursor[i]
            child[2 * i + 1] = parent[i] - cursor[i]

        gc_a = np.zeros(n_active, dtype=np.int32)
        gc_dir = np.ones(n_active, dtype=np.int32)
        gc_focus = np.arange(n_active + 1, dtype=np.int32)

        while True:
            j = gc_focus[0]
            if j == n_active:
                break
            gc_focus[0] = 0

            pos = active_pos[j]
            gc_a[j] += gc_dir[j]
            cursor[pos] = lo_arr[pos] + gc_a[j]

            if gc_a[j] == 0 or gc_a[j] == radix_arr[j] - 1:
                gc_dir[j] = -gc_dir[j]
                gc_focus[j] = gc_focus[j + 1]
                gc_focus[j + 1] = j + 1

            child[2 * pos] = cursor[pos]
            child[2 * pos + 1] = parent[pos] - cursor[pos]

            # Check subtree prune at this j
            if j >= 2 and n_active > j:
                fixed_parent_boundary = active_pos[j - 1]
                fixed_len = 2 * fixed_parent_boundary
                if fixed_len < 4:
                    continue

                total_subtree_checks += 1
                subtree_size = 1
                for kk in range(j):
                    subtree_size *= radix_arr[kk]

                # Original: j=7 only, fixed-only windows
                if j == 7:
                    if _subtree_check(child, fixed_len, d_parent, d_child, m,
                                      threshold_table, parent_prefix, lo_arr, hi_arr):
                        subtree_orig += subtree_size

                # P1: any j, fixed-only
                p1_fires = _subtree_check(child, fixed_len, d_parent, d_child, m,
                                          threshold_table, parent_prefix, lo_arr, hi_arr)
                if p1_fires:
                    subtree_p1 += subtree_size
                    total_children_p1_skip += subtree_size

                # P2: any j, with min unfixed
                p2_fires = _subtree_check(child, fixed_len, d_parent, d_child, m,
                                          threshold_table, parent_prefix, lo_arr, hi_arr,
                                          use_min_unfixed=True)
                if p2_fires:
                    subtree_p2 += subtree_size

                # P4: any j, min unfixed + partial overlap
                p4_fires = _subtree_check(child, fixed_len, d_parent, d_child, m,
                                          threshold_table, parent_prefix, lo_arr, hi_arr,
                                          use_min_unfixed=True, use_partial_overlap=True)
                if p4_fires:
                    subtree_p4 += subtree_size

        if (pidx + 1) % max(1, n_parents // 5) == 0 or pidx == n_parents - 1:
            print("  [%d/%d] tc_orig=%d, tc_p3=%d, subtree: orig=%d p1=%d p2=%d p4=%d" % (
                pidx + 1, n_parents, total_tc_orig, total_tc_p3,
                subtree_orig, subtree_p1, subtree_p2, subtree_p4))

    elapsed = time.time() - t_start
    prune_rate = 100 * (1 - total_surv_ref / total_children_ref) if total_children_ref > 0 else 0

    print("\n" + "=" * 72)
    print("RESULTS: %s (d_parent=%d, d_child=%d, %d parents, %.1fs)" % (
        level_name, d_parent, d_child, n_parents, elapsed))
    print("=" * 72)
    print("  Reference prune rate: %.2f%%" % prune_rate)
    print("  Total children (reference): %d" % total_children_ref)
    print("  Total survivors: %d" % total_surv_ref)
    print()
    print("  PROPOSAL 3 (Arc Consistency, W_int_max):")
    print("    Original Cartesian product: %d" % total_tc_orig)
    print("    After tightening:           %d" % total_tc_p3)
    if total_tc_orig > 0:
        print("    Reduction: %.2fx" % (total_tc_orig / max(1, total_tc_p3)))
    print()
    print("  SUBTREE PRUNING (children skippable):")
    print("    Original (j=7 only, fixed windows): %d" % subtree_orig)
    print("    P1 (j>=2, fixed windows):           %d" % subtree_p1)
    print("    P2 (j>=2, + min unfixed):           %d" % subtree_p2)
    print("    P4 (j>=2, + min unfixed + overlap): %d" % subtree_p4)
    print()

    if subtree_orig > 0:
        print("    P1 vs original: %.1fx" % (subtree_p1 / subtree_orig))
    if subtree_p1 > 0:
        print("    P2 vs P1: %.1fx" % (subtree_p2 / subtree_p1))
        print("    P4 vs P1: %.1fx" % (subtree_p4 / subtree_p1))
    if total_children_ref > 0 and subtree_p4 > 0:
        effective_tested = total_children_ref - subtree_p4
        print()
        print("  COMBINED EFFECT:")
        print("    Children after P3 range tightening: %d" % total_tc_p3)
        print("    Subtree skip (P1+P2+P4): %d children" % subtree_p4)
        print("    Effective children to test: ~%d" % max(0, total_tc_p3 - subtree_p4))
        if total_children_ref > 0:
            combined_reduction = total_children_ref / max(1, max(0, total_tc_p3 - subtree_p4))
            print("    Combined reduction vs reference: %.1fx" % combined_reduction)

    return {
        'tc_orig': total_tc_orig,
        'tc_p3': total_tc_p3,
        'subtree_orig': subtree_orig,
        'subtree_p1': subtree_p1,
        'subtree_p2': subtree_p2,
        'subtree_p4': subtree_p4,
        'children_ref': total_children_ref,
        'survivors': total_surv_ref,
    }


def generate_test_data():
    """Generate test parents at levels where pruning fires."""
    m = 15; n_half = 3; c_target = 1.33

    print("Generating test data with m=%d, n_half=%d, c_target=%.2f" % (m, n_half, c_target))

    l0 = _cascade.run_level0(n_half, m, c_target, verbose=False)
    print("L0: %d survivors" % l0['n_survivors'])

    current = l0['survivors']
    saved = {}
    for level in range(1, 6):
        if len(current) == 0:
            print("L%d: PROVEN" % level)
            break

        d_child = 2 * current.shape[1]
        n_half_child = n_half * (2 ** level)

        n_proc = min(len(current), 1000)
        children = []
        total_tc = 0
        total_surv = 0
        for i in range(n_proc):
            surv, tc = process_parent_fused(current[i], m, c_target, n_half_child)
            total_tc += tc
            total_surv += len(surv)
            if len(surv) > 0:
                children.append(surv)

        prune_pct = 100 * (1 - total_surv / total_tc) if total_tc > 0 else 0
        print("L%d: %d parents, tc=%d, surv=%d, prune=%.1f%%" % (
            level, len(current), total_tc, total_surv, prune_pct))

        if prune_pct > 10:
            saved[level] = current[:n_proc].copy()

        if children:
            current = np.unique(np.concatenate(children), axis=0)
        else:
            current = np.empty((0, d_child), dtype=np.int32)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, help="Test only this level")
    parser.add_argument("--max_parents", type=int, default=100)
    args = parser.parse_args()

    m = 15; n_half = 3; c_target = 1.33

    # Try to load pre-generated test data
    levels_to_test = {}
    for lvl in [1, 2, 3, 4]:
        path = os.path.join(_repo_dir, "data", "pruning_test_L%d.npy" % lvl)
        if os.path.exists(path):
            levels_to_test[lvl] = np.load(path)

    if not levels_to_test:
        print("No test data found. Generating...")
        levels_to_test = generate_test_data()

    if args.level:
        if args.level in levels_to_test:
            levels_to_test = {args.level: levels_to_test[args.level]}
        else:
            print("Level %d not available. Available: %s" % (
                args.level, sorted(levels_to_test.keys())))
            return

    all_results = {}
    for lvl in sorted(levels_to_test.keys()):
        parents = levels_to_test[lvl]
        n_half_child = n_half * (2 ** lvl)
        print("\n" + "#" * 72)
        print("LEVEL %d: %d parents, d_parent=%d, d_child=%d, n_half_child=%d" % (
            lvl, len(parents), parents.shape[1], 2 * parents.shape[1], n_half_child))
        print("#" * 72)

        r = bench_proposals(parents, m, c_target, n_half_child,
                            max_parents=args.max_parents,
                            level_name="L%d" % lvl)
        all_results[lvl] = r

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("%-6s %12s %12s %12s %12s %12s %12s" % (
        "Level", "Children", "P3 tight", "P3 red.",
        "Subtree(j=7)", "Subtree(all)", "P4(all)"))
    for lvl in sorted(all_results.keys()):
        r = all_results[lvl]
        p3_red = r['tc_orig'] / max(1, r['tc_p3'])
        print("%-6s %12d %12d %11.1fx %12d %12d %12d" % (
            "L%d" % lvl, r['children_ref'], r['tc_p3'], p3_red,
            r['subtree_orig'], r['subtree_p1'], r['subtree_p4']))


if __name__ == "__main__":
    main()
