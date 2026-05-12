"""Full Parent-Level Min-Contribution Pruning — cascade test.

Starts from L0 survivors (d=4), runs the Gray code kernel on random
subsets at each level to collect survivors, then measures the
min-contribution pruning rate at each dimension as d doubles.

Parameters: c_target=1.35, m=20, d0=2.
"""
import os
import sys
import time

import numpy as np
from numba import njit

# ── path setup ──────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_root, "cloninger-steinerberger")
sys.path.insert(0, _cs_dir)
sys.path.insert(0, _root)

from cpu.run_cascade import (
    _compute_bin_ranges,
    _tighten_ranges,
    _fused_generate_and_prune_gray,
    process_parent_fused,
)


# =====================================================================
# Core: parent-level min-contribution pruning
# =====================================================================

@njit(cache=True)
def _parent_min_contribution_pruned(parent_int, lo_arr, hi_arr,
                                     m, c_target, n_half_child,
                                     use_flat_threshold=False):
    """Return True if parent is provably dead (all children will be pruned).

    Computes a guaranteed lower bound on conv[k] for every k by using the
    minimum possible value of each child bin.  For the mutual term (bins
    from the same parent), uses the endpoint minimum of the concave
    parabola 2*v*(2B-v).  For cross-terms between different parents, uses
    the independent minimums.

    Threshold uses W_int_max (maximum possible mass in window) to get the
    highest (most generous) threshold.  If min_ws > threshold(ell, W_int_max)
    for ANY window, the parent is dead.
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    # --- Build child_min[i] for all d_child bins ---
    child_min = np.empty(d_child, dtype=np.int64)
    for p in range(d_parent):
        child_min[2 * p] = np.int64(lo_arr[p])
        child_min[2 * p + 1] = np.int64(2 * parent_int[p] - hi_arr[p])

    # --- Compute min_conv[k] ---
    min_conv = np.zeros(conv_len, dtype=np.int64)

    for p in range(d_parent):
        k1 = 2 * p
        k2 = 2 * p + 1
        ml = np.int64(lo_arr[p])
        mh = np.int64(2 * parent_int[p] - hi_arr[p])

        # Self-terms
        min_conv[2 * k1] += ml * ml
        min_conv[2 * k2] += mh * mh

        # Mutual term: 2*v*(2B-v) is concave in v, min at endpoint
        lo_val = np.int64(lo_arr[p])
        hi_val = np.int64(hi_arr[p])
        two_P = np.int64(2) * np.int64(parent_int[p])
        mut_lo = np.int64(2) * lo_val * (two_P - lo_val)
        mut_hi = np.int64(2) * hi_val * (two_P - hi_val)
        min_conv[k1 + k2] += mut_lo if mut_lo < mut_hi else mut_hi

    # Cross-terms between different parents
    for p in range(d_parent):
        k1p = 2 * p
        k2p = 2 * p + 1
        ml_p = child_min[k1p]
        mh_p = child_min[k2p]
        for q in range(p + 1, d_parent):
            k1q = 2 * q
            k2q = 2 * q + 1
            ml_q = child_min[k1q]
            mh_q = child_min[k2q]
            if ml_p > 0 and ml_q > 0:
                min_conv[k1p + k1q] += np.int64(2) * ml_p * ml_q
            if ml_p > 0 and mh_q > 0:
                min_conv[k1p + k2q] += np.int64(2) * ml_p * mh_q
            if mh_p > 0 and ml_q > 0:
                min_conv[k2p + k1q] += np.int64(2) * mh_p * ml_q
            if mh_p > 0 and mh_q > 0:
                min_conv[k2p + k2q] += np.int64(2) * mh_p * mh_q

    # --- Max child prefix sum for W_int_max queries ---
    max_child_prefix = np.empty(d_child + 1, dtype=np.int64)
    max_child_prefix[0] = np.int64(0)
    for q in range(d_parent):
        max_child_prefix[2 * q + 1] = (max_child_prefix[2 * q]
                                        + np.int64(hi_arr[q]))
        max_child_prefix[2 * q + 2] = (max_child_prefix[2 * q + 1]
                                        + np.int64(2 * parent_int[q] - lo_arr[q]))

    # --- Precompute threshold table ---
    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half_child)
    n_half_d = np.float64(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    S_child_plus_1 = int(4 * n_half_child * m + 1)
    ell_count = conv_len
    flat_corr = 2.0 * m_d + 1.0

    threshold_table = np.empty(ell_count * S_child_plus_1, dtype=np.int64)
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

    # --- Window scan over min_conv ---
    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        ell_idx = ell - 2

        ws = np.int64(0)
        for kk in range(n_cv):
            ws += min_conv[kk]

        hb = ell - 2
        if hb > d_child - 1:
            hb = d_child - 1
        W_max = max_child_prefix[hb + 1]
        if W_max >= np.int64(S_child_plus_1):
            W_max = np.int64(S_child_plus_1 - 1)
        if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
            return True

        for s in range(1, n_windows):
            ws += min_conv[s + n_cv - 1] - min_conv[s - 1]
            lb = s - (d_child - 1)
            if lb < 0:
                lb = 0
            hb = s + ell - 2
            if hb > d_child - 1:
                hb = d_child - 1
            W_max = max_child_prefix[hb + 1] - max_child_prefix[lb]
            if W_max >= np.int64(S_child_plus_1):
                W_max = np.int64(S_child_plus_1 - 1)
            if ws > threshold_table[ell_idx * S_child_plus_1 + int(W_max)]:
                return True

    return False


# =====================================================================
# Helpers
# =====================================================================

THROUGHPUT = 7_000_000  # children/sec/core


def measure_min_contribution(parents, m, c_target, use_flat, level_label):
    """Run min-contribution check on all parents, print stats, return dict."""
    n_parents = parents.shape[0]
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    pruned_count = 0
    live_count = 0
    empty_count = 0
    children_pruned = 0
    children_live = 0
    timings_ns = []
    pruned_small = []   # (idx, tc) for pruned with tc < 500K
    live_small = []     # (idx, tc) for live with tc < 500K

    for i in range(n_parents):
        parent = parents[i]
        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            empty_count += 1
            continue
        lo, hi, _ = result
        tc = _tighten_ranges(parent, lo, hi, m, c_target, n_half_child, use_flat)
        if tc == 0:
            empty_count += 1
            continue

        t0 = time.perf_counter_ns()
        is_pruned = _parent_min_contribution_pruned(
            parent, lo, hi, m, c_target, n_half_child, use_flat)
        t1 = time.perf_counter_ns()
        timings_ns.append(t1 - t0)

        if is_pruned:
            pruned_count += 1
            children_pruned += tc
            if tc < 500_000:
                pruned_small.append((i, int(tc)))
        else:
            live_count += 1
            children_live += tc
            if tc < 500_000:
                live_small.append((i, int(tc)))

    active = pruned_count + live_count
    rate = pruned_count / active * 100 if active > 0 else 0
    saved_s = children_pruned / THROUGHPUT
    t_arr = np.array(timings_ns, dtype=np.float64) if timings_ns else np.array([0.0])

    print(f"\n  [{level_label}] d_parent={d_parent}, d_child={d_child}")
    print(f"    Total parents:        {n_parents}")
    print(f"    Empty-range:          {empty_count}")
    print(f"    Active parents:       {active}")
    print(f"    Min-contrib pruned:   {pruned_count}  ({rate:.2f}%)")
    print(f"    Live (not pruned):    {live_count}")
    print(f"    Children pruned:      {children_pruned:,.0f}")
    print(f"    Children live:        {children_live:,.0f}")
    print(f"    Est. time saved:      {saved_s:.1f}s ({saved_s/3600:.3f} hrs)")
    print(f"    Check cost (ns):      mean={t_arr.mean():.0f}  "
          f"p50={np.percentile(t_arr, 50):.0f}  max={t_arr.max():.0f}")

    return {
        "pruned_count": pruned_count, "live_count": live_count,
        "active": active, "rate": rate,
        "children_pruned": children_pruned, "children_live": children_live,
        "saved_s": saved_s, "timings": t_arr,
        "pruned_small": pruned_small, "live_small": live_small,
    }


def collect_survivors(parents, m, c_target, n_half_child, use_flat,
                      max_parents, max_time_per_parent, max_survivors,
                      max_surv_per_parent, label):
    """Run Gray code kernel on a random subset of parents, collect survivors.

    Picks parents randomly, skips those whose tc exceeds the time budget.
    Caps survivors per parent (for diversity) and total (for OOM safety).
    Returns survivors array (n_surv, d_child).
    """
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_total = parents.shape[0]

    # Random permutation of parent indices
    rng = np.random.default_rng(42)
    order = rng.permutation(n_total)

    print(f"\n  [{label}] Collecting survivors from {n_total} parents "
          f"(max {max_parents} runs, max {max_survivors:,} total, "
          f"max {max_surv_per_parent:,}/parent)")

    all_survivors = []
    total_time = 0.0
    total_children_tested = 0
    n_run = 0
    n_skipped = 0
    n_surv_total = 0

    for oi in range(n_total):
        if n_run >= max_parents or n_surv_total >= max_survivors:
            break

        idx = order[oi]
        parent = parents[idx]
        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            continue
        lo, hi, _ = result
        tc = _tighten_ranges(parent, lo, hi, m, c_target, n_half_child, use_flat)
        if tc <= 0:
            continue

        est_s = tc / THROUGHPUT
        if est_s > max_time_per_parent:
            n_skipped += 1
            continue

        t0 = time.time()
        survivors, tc_actual = process_parent_fused(
            parent, m, c_target, n_half_child, use_flat_threshold=use_flat)
        elapsed = time.time() - t0
        total_time += elapsed
        total_children_tested += tc_actual
        n_run += 1

        if survivors.shape[0] > 0:
            # Cap per-parent for diversity, then cap total for OOM
            keep = min(survivors.shape[0], max_surv_per_parent,
                       max_survivors - n_surv_total)
            if keep > 0:
                # Random sample if we're capping
                if keep < survivors.shape[0]:
                    idx_s = rng.choice(survivors.shape[0], keep, replace=False)
                    all_survivors.append(survivors[idx_s])
                else:
                    all_survivors.append(survivors)
                n_surv_total += keep

        if n_run % max(1, max_parents // 5) == 0 or n_run == max_parents:
            print(f"    {n_run}/{max_parents}: {total_children_tested:,.0f} children, "
                  f"{n_surv_total:,} survivors, {total_time:.1f}s "
                  f"({n_skipped} skipped)")

    if all_survivors:
        result_arr = np.concatenate(all_survivors, axis=0)
    else:
        result_arr = np.empty((0, d_child), dtype=np.int32)

    print(f"    Done: {result_arr.shape[0]:,} survivors from "
          f"{total_children_tested:,.0f} children ({n_run} parents, "
          f"{n_skipped} skipped) in {total_time:.1f}s")
    return result_arr


def validate_pruned(parents, pruned_indices, m, c_target, n_half_child,
                    use_flat, n_validate, label):
    """Run kernel on pruned parents, verify 0 survivors. Return pass/fail."""
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    all_pass = True

    n_val = min(n_validate, len(pruned_indices))
    if n_val == 0:
        print(f"  [{label}] No pruned parents to validate")
        return True

    print(f"  [{label}] Validating {n_val} pruned parents (expect 0 survivors each):")
    for vi in range(n_val):
        idx, tc = pruned_indices[vi]
        parent = parents[idx]
        survivors, _ = process_parent_fused(
            parent, m, c_target, n_half_child, use_flat_threshold=use_flat)
        ok = survivors.shape[0] == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"    parent[{idx}] d={d_parent} tc={tc:,}  "
              f"survivors={survivors.shape[0]}  {status}")
    return all_pass


# =====================================================================
# Main
# =====================================================================

def main():
    m = 20
    c_target = 1.40
    use_flat = False

    data_path = os.path.join(_root, "data", "checkpoint_L0_survivors.npy")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found")
        return
    l0_parents = np.load(data_path)
    print(f"Config: m={m}, c_target={c_target}, d0=2, use_flat={use_flat}")
    print(f"Loaded {l0_parents.shape[0]} L0 survivors (d={l0_parents.shape[1]})")

    # ── JIT warmup ──────────────────────────────────────────────────────
    print("\nWarming up JIT (first call compiles)...")
    dummy = l0_parents[0].copy()
    d_child = 2 * dummy.shape[0]
    nhc = d_child // 2
    result = _compute_bin_ranges(dummy, m, c_target, d_child, nhc)
    if result is not None:
        lo, hi, _ = result
        _tighten_ranges(dummy, lo, hi, m, c_target, nhc, use_flat)
        _parent_min_contribution_pruned(dummy, lo, hi, m, c_target, nhc, use_flat)
    # Warmup the kernel too
    _ = process_parent_fused(dummy, m, c_target, nhc, use_flat_threshold=use_flat)
    print("JIT warm-up done.")

    # ── Cascade: collect survivors at each level, measure pruning ───────
    # L0 survivors (d=4) are parents for L1 (d_child=8)
    # L1 survivors (d=8) are parents for L2 (d_child=16)
    # etc.

    all_pass = True
    level_parents = {
        "L1": l0_parents,  # d_parent=4, d_child=8
    }

    # Time budgets per level:
    #   (max_time_per_parent_s, max_parents_to_run, max_survivors_to_keep,
    #    max_surv_per_parent)
    budgets = {
        "L1": (30.0, 300, 500_000, 2000),
        "L2": (120.0, 200, 100_000, 1000),
        "L3": (300.0, 100, 50_000, 500),
        "L4": (300.0, 50, 10_000, 200),
    }
    # Max parents to measure min-contribution on (sample if more)
    MAX_MEASURE = 200_000

    levels = ["L1", "L2", "L3", "L4"]
    results = {}

    for level in levels:
        if level not in level_parents:
            print(f"\n{'='*72}")
            print(f"  {level}: No parents available (previous level had 0 survivors)")
            break

        parents = level_parents[level]
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        print(f"\n{'='*72}")
        print(f"{level}: d_parent={d_parent} -> d_child={d_child}  "
              f"({parents.shape[0]} parents)")
        print(f"{'='*72}")

        # ── Measure min-contribution pruning (sample if too many) ──
        if parents.shape[0] > MAX_MEASURE:
            rng = np.random.default_rng(123)
            sample_idx = rng.choice(parents.shape[0], MAX_MEASURE, replace=False)
            measure_parents = parents[sample_idx]
            print(f"  (Sampling {MAX_MEASURE:,} of {parents.shape[0]:,} parents)")
        else:
            measure_parents = parents
        stats = measure_min_contribution(measure_parents, m, c_target, use_flat, level)
        results[level] = stats

        # ── Validate pruned parents (soundness check) ──
        if stats["pruned_small"]:
            ok = validate_pruned(measure_parents, stats["pruned_small"],
                                 m, c_target, n_half_child, use_flat,
                                 n_validate=3, label=level)
            if not ok:
                all_pass = False

        # ── Collect survivors to use as parents for next level ──
        next_level_idx = levels.index(level) + 1
        if next_level_idx < len(levels):
            next_level = levels[next_level_idx]
            max_time, max_par, max_surv, max_spp = budgets[level]
            survivors = collect_survivors(
                parents, m, c_target, n_half_child, use_flat,
                max_parents=max_par, max_time_per_parent=max_time,
                max_survivors=max_surv, max_surv_per_parent=max_spp,
                label=f"{level}->survivors")

            if survivors.shape[0] > 0:
                level_parents[next_level] = survivors
                print(f"  -> {survivors.shape[0]} survivors become "
                      f"{next_level} parents (d={survivors.shape[1]})")
            else:
                print(f"  -> 0 survivors, cascade stops here")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY TABLE")
    print(f"{'='*72}")
    print(f"  {'Level':<6} {'d_par':>5} {'d_chi':>5} {'Parents':>8} "
          f"{'Pruned':>8} {'Rate':>7} {'Children saved':>18} {'Time saved':>12}")
    print(f"  {'-'*6} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*18} {'-'*12}")
    for level in levels:
        if level not in results:
            break
        s = results[level]
        d_par = level_parents[level].shape[1]
        d_chi = 2 * d_par
        print(f"  {level:<6} {d_par:>5} {d_chi:>5} {s['active']:>8} "
              f"{s['pruned_count']:>8} {s['rate']:>6.1f}% "
              f"{s['children_pruned']:>18,} {s['saved_s']:>10.1f}s")

    print(f"\n  Correctness: {'ALL PASS' if all_pass else 'ISSUES FOUND'}")
    print()


if __name__ == "__main__":
    main()
