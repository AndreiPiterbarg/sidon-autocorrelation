#!/usr/bin/env python3
"""Test pruning ideas and measure their impact on L0->L1 and L1->L2.

Three ideas tested:
  1. Separable min-ws parent kill: lower-bound the minimum window sum over
     ALL cursor assignments.  If it exceeds the threshold, the parent is
     killed without enumerating any children.
  2. Cursor range tightening (_tighten_ranges): already implemented in
     run_cascade.py.  Measure how many parents it kills and how many
     children it saves.
  3. Increased m: run L0->L1 with m=20,30,40,50 and compare survivors.

Usage:
    python tests/test_pruning_ideas.py
    python tests/test_pruning_ideas.py --c_target 1.40
"""
import argparse
import math
import os
import sys
import time

import numpy as np

_cs_root = os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger')
_cs_cpu = os.path.join(_cs_root, 'cpu')
sys.path.insert(0, os.path.abspath(_cs_root))
sys.path.insert(0, os.path.abspath(_cs_cpu))

from pruning import correction, count_compositions, asymmetry_threshold
from run_cascade import (
    run_level0, _compute_bin_ranges,
    _tighten_ranges, _fused_generate_and_prune_gray,
    _canonicalize_inplace, _fast_dedup, _default_buf_cap,
)


# ================================================================
# IDEA 1: Separable min-ws parent kill
# ================================================================

def _self_term_min(p_i, lo_i, hi_i, I):
    """Minimum of self-term contribution over cursor range [lo_i, hi_i].

    Self-term for parent position i with child bins {2i, 2i+1}:
      conv[4i]   += c^2           (I[0])
      conv[4i+1] += 2c(2p-c)     (I[1])
      conv[4i+2] += (2p-c)^2     (I[2])

    self(c) = I[0]*c^2 + I[1]*2c(2p-c) + I[2]*(2p-c)^2
            = A*c^2 + B*c + C

    Returns the minimum value over c in [lo_i, hi_i].
    """
    A = float(I[0] - 2 * I[1] + I[2])
    B = 4.0 * p_i * float(I[1] - I[2])
    C = 4.0 * p_i * p_i * float(I[2])

    v_lo = A * lo_i * lo_i + B * lo_i + C
    v_hi = A * hi_i * hi_i + B * hi_i + C
    v_min = min(v_lo, v_hi)

    if A > 1e-15:
        # Convex parabola — check vertex
        c_star = -B / (2.0 * A)
        if lo_i <= c_star <= hi_i:
            v_star = A * c_star * c_star + B * c_star + C
            v_min = min(v_min, v_star)

    return v_min


def _cross_term_min(p_i, p_j, lo_i, hi_i, lo_j, hi_j, J):
    """Minimum of cross-term contribution over cursor box.

    Cross-term for parent pair (i,j) with i<j:
      conv[2(i+j)]   += 2*c_i*c_j                                (J[0])
      conv[2(i+j)+1] += 2*c_i*(2p_j-c_j) + 2*(2p_i-c_i)*c_j    (J[1])
      conv[2(i+j)+2] += 2*(2p_i-c_i)*(2p_j-c_j)                 (J[2])

    cross(c_i, c_j) = alpha*c_i*c_j + beta*c_i + gamma*c_j + delta

    Bilinear in (c_i, c_j) => minimum at one of 4 box corners.
    """
    alpha = 2.0 * J[0] - 4.0 * J[1] + 2.0 * J[2]
    beta = 4.0 * p_j * float(J[1] - J[2])
    gamma = 4.0 * p_i * float(J[1] - J[2])
    delta = 8.0 * p_i * p_j * float(J[2])

    vals = []
    for ci in (lo_i, hi_i):
        for cj in (lo_j, hi_j):
            vals.append(alpha * ci * cj + beta * ci + gamma * cj + delta)

    return min(vals)


def separable_min_ws(parent_int, m, c_target, n_half_child):
    """Separable lower bound on min window sum over all cursor assignments.

    For each window (ell, s_lo), decomposes ws into independent self-terms
    (per parent position) and cross-terms (per parent pair), minimizes
    each independently.  Sum of minimums <= min of sum (valid lower bound).

    Returns (killed, best_gap, best_window) where:
      killed: True if some window's min-ws exceeds its threshold
      best_gap: tightest (ws_min - threshold) found
      best_window: (ell, s_lo) of tightest gap
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    result = _compute_bin_ranges(parent_int, m, c_target, d_child, n_half_child)
    if result is None:
        return True, float('inf'), (0, 0)
    lo_arr, hi_arr, total_children = result
    if total_children == 0:
        return True, float('inf'), (0, 0)

    m_d = float(m)
    n_half_d = float(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    four_n = 4.0 * n_half_d
    S_child = int(4 * n_half_child * m)

    # Max child values for W_int_max
    max_child = np.empty(d_child, dtype=np.float64)
    for i in range(d_parent):
        max_child[2 * i] = float(hi_arr[i])
        max_child[2 * i + 1] = float(2 * parent_int[i] - lo_arr[i])

    p = parent_int.astype(np.float64)
    lo = lo_arr.astype(np.float64)
    hi = hi_arr.astype(np.float64)

    best_gap = -float('inf')
    best_window = (0, 0)

    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1

        for s_lo in range(n_windows):
            s_hi = s_lo + n_cv - 1

            ws_min = 0.0

            # Self terms: parent position i -> conv positions {4i, 4i+1, 4i+2}
            for i in range(d_parent):
                I = [0, 0, 0]
                for t in range(3):
                    pos = 4 * i + t
                    if s_lo <= pos <= s_hi:
                        I[t] = 1

                if I[0] == 0 and I[1] == 0 and I[2] == 0:
                    continue

                ws_min += _self_term_min(p[i], lo[i], hi[i], I)

            # Cross terms: pair (i,j) -> conv positions {2(i+j), 2(i+j)+1, 2(i+j)+2}
            for i in range(d_parent):
                for j in range(i + 1, d_parent):
                    base = 2 * (i + j)
                    J = [0, 0, 0]
                    for t in range(3):
                        pos = base + t
                        if s_lo <= pos <= s_hi:
                            J[t] = 1

                    if J[0] == 0 and J[1] == 0 and J[2] == 0:
                        continue

                    ws_min += _cross_term_min(
                        p[i], p[j], lo[i], hi[i], lo[j], hi[j], J)

            # W_int_max for threshold (cap at S_child)
            lo_bin = max(0, s_lo - (d_child - 1))
            hi_bin = min(d_child - 1, s_lo + ell - 2)
            W_int_max = sum(max_child[k] for k in range(lo_bin, hi_bin + 1))
            W_int_max = min(W_int_max, float(S_child))

            corr_w = 1.0 + W_int_max / (2.0 * n_half_d)
            threshold = math.floor(
                (cs_base_m2 + corr_w + eps_margin) * four_n * float(ell))

            gap = ws_min - threshold
            if gap > best_gap:
                best_gap = gap
                best_window = (ell, s_lo)

            if ws_min > threshold:
                return True, gap, (ell, s_lo)

    return False, best_gap, best_window


# ================================================================
# IDEA 2: Measure _tighten_ranges impact
# ================================================================

def measure_tightening_impact(parents, m, c_target, n_half_child):
    """For each parent, measure how much _tighten_ranges helps.

    Returns dict with:
      parents_killed: # parents where tightening reduces to 0 children
      total_original: sum of original Cartesian product sizes
      total_tightened: sum of tightened Cartesian product sizes
      avg_reduction_pct: average % reduction in children
      per_parent: list of (original, tightened) per parent
    """
    d_child = 2 * parents.shape[1]
    parents_killed = 0
    total_original = 0
    total_tightened = 0
    per_parent = []

    for idx in range(len(parents)):
        parent_int = parents[idx]

        result = _compute_bin_ranges(
            parent_int, m, c_target, d_child, n_half_child)
        if result is None:
            per_parent.append((0, 0))
            parents_killed += 1
            continue

        lo_arr, hi_arr, original = result
        total_original += original

        lo_copy = lo_arr.copy()
        hi_copy = hi_arr.copy()
        tightened = _tighten_ranges(
            parent_int, lo_copy, hi_copy, m, c_target, n_half_child)

        total_tightened += tightened
        per_parent.append((original, tightened))
        if tightened == 0:
            parents_killed += 1

    reduction_pcts = []
    for orig, tight in per_parent:
        if orig > 0:
            reduction_pcts.append(100.0 * (1.0 - tight / orig))

    return {
        'parents_killed': parents_killed,
        'total_parents': len(parents),
        'total_original': total_original,
        'total_tightened': total_tightened,
        'overall_reduction_pct': (
            100.0 * (1.0 - total_tightened / total_original)
            if total_original > 0 else 0.0),
        'avg_reduction_pct': (
            np.mean(reduction_pcts) if reduction_pcts else 0.0),
        'per_parent': per_parent,
    }


def run_level_no_tighten(parents, m, c_target, n_half_child):
    """Run one cascade level WITHOUT _tighten_ranges.

    Uses the Gray code kernel directly with original (un-tightened) ranges.
    Returns (total_survivors, total_children).
    """
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    total_survivors = 0
    total_children = 0
    all_survivors = []

    for idx in range(len(parents)):
        parent_int = parents[idx]
        result = _compute_bin_ranges(
            parent_int, m, c_target, d_child, n_half_child)
        if result is None:
            continue

        lo_arr, hi_arr, n_children = result
        total_children += n_children
        if n_children == 0:
            continue

        buf_cap = min(n_children, _default_buf_cap(d_child))
        out_buf = np.empty((buf_cap, d_child), dtype=np.int32)
        n_surv, _ = _fused_generate_and_prune_gray(
            parent_int, n_half_child, m, c_target,
            lo_arr, hi_arr, out_buf)

        if n_surv > buf_cap:
            out_buf = np.empty((n_surv, d_child), dtype=np.int32)
            n_surv, _ = _fused_generate_and_prune_gray(
                parent_int, n_half_child, m, c_target,
                lo_arr, hi_arr, out_buf)

        if n_surv > 0:
            all_survivors.append(out_buf[:n_surv].copy())
        total_survivors += n_surv

    if all_survivors:
        survivors = np.vstack(all_survivors)
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)

    return survivors, total_children


def run_level_with_tighten(parents, m, c_target, n_half_child):
    """Run one cascade level WITH _tighten_ranges (the normal path).

    Returns (total_survivors, total_children_tightened).
    """
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    total_survivors = 0
    total_children = 0
    all_survivors = []

    for idx in range(len(parents)):
        parent_int = parents[idx]
        result = _compute_bin_ranges(
            parent_int, m, c_target, d_child, n_half_child)
        if result is None:
            continue

        lo_arr, hi_arr, n_ch_orig = result

        # Apply tightening
        n_ch = _tighten_ranges(
            parent_int, lo_arr, hi_arr, m, c_target, n_half_child)
        total_children += n_ch

        if n_ch == 0:
            continue

        buf_cap = min(n_ch, _default_buf_cap(d_child))
        out_buf = np.empty((buf_cap, d_child), dtype=np.int32)
        n_surv, _ = _fused_generate_and_prune_gray(
            parent_int, n_half_child, m, c_target,
            lo_arr, hi_arr, out_buf)

        if n_surv > buf_cap:
            out_buf = np.empty((n_surv, d_child), dtype=np.int32)
            n_surv, _ = _fused_generate_and_prune_gray(
                parent_int, n_half_child, m, c_target,
                lo_arr, hi_arr, out_buf)

        if n_surv > 0:
            all_survivors.append(out_buf[:n_surv].copy())
        total_survivors += n_surv

    if all_survivors:
        survivors = np.vstack(all_survivors)
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)

    return survivors, total_children


def dedup_survivors(survivors):
    """Canonicalize and deduplicate survivor array."""
    if len(survivors) == 0:
        return survivors
    _canonicalize_inplace(survivors)
    return _fast_dedup(survivors)


# ================================================================
# Test runners
# ================================================================

def test_idea1(parents, m, c_target, n_half_child, label=""):
    """Test Idea 1: separable min-ws parent kill."""
    print(f"\n{'='*70}")
    print(f"IDEA 1: Separable Min-WS Parent Kill  {label}")
    print(f"{'='*70}")
    print(f"  Parents: {len(parents)}, d_parent={parents.shape[1]}, "
          f"d_child={2*parents.shape[1]}")
    print(f"  m={m}, c_target={c_target}, n_half_child={n_half_child}")

    n_killed = 0
    n_total = len(parents)
    gaps = []
    t0 = time.time()

    for idx in range(n_total):
        parent_int = parents[idx]
        killed, gap, window = separable_min_ws(
            parent_int, m, c_target, n_half_child)
        gaps.append(gap)
        if killed:
            n_killed += 1
            print(f"    Parent {idx} KILLED by window ell={window[0]}, "
                  f"s_lo={window[1]}, gap={gap:.1f}")

    elapsed = time.time() - t0
    gaps = np.array(gaps)

    print(f"\n  Results ({elapsed:.2f}s):")
    print(f"    Parents killed: {n_killed}/{n_total} "
          f"({100*n_killed/max(1,n_total):.1f}%)")
    print(f"    Best gap (ws_min - threshold): "
          f"min={gaps.min():.1f}, median={np.median(gaps):.1f}, "
          f"max={gaps.max():.1f}")
    if gaps.max() < 0:
        print(f"    Closest to killing: gap={gaps.max():.1f} "
              f"(need > 0 to kill)")
    return n_killed


def test_idea2(parents, m, c_target, n_half_child, label=""):
    """Test Idea 2: cursor range tightening impact."""
    print(f"\n{'='*70}")
    print(f"IDEA 2: Cursor Range Tightening Impact  {label}")
    print(f"{'='*70}")
    print(f"  Parents: {len(parents)}, d_parent={parents.shape[1]}, "
          f"d_child={2*parents.shape[1]}")
    print(f"  m={m}, c_target={c_target}, n_half_child={n_half_child}")

    # Phase 1: measure tightening statistics
    t0 = time.time()
    stats = measure_tightening_impact(parents, m, c_target, n_half_child)
    t_measure = time.time() - t0

    print(f"\n  Tightening statistics ({t_measure:.2f}s):")
    print(f"    Parents killed by tightening: {stats['parents_killed']}"
          f"/{stats['total_parents']}")
    print(f"    Total children (original):  {stats['total_original']:,}")
    print(f"    Total children (tightened): {stats['total_tightened']:,}")
    print(f"    Overall reduction: {stats['overall_reduction_pct']:.1f}%")
    print(f"    Avg per-parent reduction: {stats['avg_reduction_pct']:.1f}%")

    # Phase 2: run cascade with vs without tightening
    print(f"\n  Running cascade WITHOUT tightening...")
    t0 = time.time()
    surv_no, ch_no = run_level_no_tighten(
        parents, m, c_target, n_half_child)
    surv_no = dedup_survivors(surv_no)
    t_no = time.time() - t0
    print(f"    {len(surv_no):,} unique survivors from {ch_no:,} children "
          f"({t_no:.2f}s)")

    print(f"  Running cascade WITH tightening...")
    t0 = time.time()
    surv_yes, ch_yes = run_level_with_tighten(
        parents, m, c_target, n_half_child)
    surv_yes = dedup_survivors(surv_yes)
    t_yes = time.time() - t0
    print(f"    {len(surv_yes):,} unique survivors from {ch_yes:,} children "
          f"({t_yes:.2f}s)")

    # Verify: tightening should not change survivor set
    if len(surv_yes) != len(surv_no):
        print(f"  WARNING: survivor count mismatch! "
              f"no_tight={len(surv_no)} vs tight={len(surv_yes)}")
    else:
        print(f"  Survivor counts match (correctness OK)")

    saved = ch_no - ch_yes
    print(f"\n  Summary:")
    print(f"    Children saved by tightening: {saved:,} "
          f"({100*saved/max(1,ch_no):.1f}%)")
    print(f"    Time saved: {t_no - t_yes:.2f}s "
          f"({100*(t_no-t_yes)/max(0.001,t_no):.1f}%)")

    return surv_yes


def test_idea3(c_target, m_values=(20, 30, 40, 50)):
    """Test Idea 3: increasing m."""
    print(f"\n{'='*70}")
    print(f"IDEA 3: Varying Grid Resolution m")
    print(f"{'='*70}")
    print(f"  c_target={c_target}")
    print(f"  m values: {m_values}")

    results = []

    for m in m_values:
        n_half = 1
        d0 = 2
        corr = correction(m, n_half)
        S = int(2 * d0 * m)

        print(f"\n  --- m={m}: correction={corr:.6f}, "
              f"threshold={c_target+corr:.6f}, S={S} ---")

        # L0
        t0 = time.time()
        l0 = run_level0(n_half, m, c_target, verbose=False, d0=d0)
        t_l0 = time.time() - t0
        l0_surv = l0['survivors']
        print(f"    L0: {l0['n_survivors']:,} survivors ({t_l0:.2f}s)")

        if l0['n_survivors'] == 0:
            print(f"    PROVEN at L0!")
            results.append({
                'm': m, 'corr': corr, 'l0_surv': 0,
                'l1_surv': 0, 'l1_children': 0,
            })
            continue

        # L1
        d_child = 2 * d0
        n_half_child = d0  # n_half doubles: n_half * 2^1 = 1*2 = 2 = d0
        t0 = time.time()
        surv_l1, ch_l1 = run_level_with_tighten(
            l0_surv, m, c_target, n_half_child)
        surv_l1 = dedup_survivors(surv_l1)
        t_l1 = time.time() - t0
        print(f"    L1: {len(surv_l1):,} unique survivors "
              f"from {ch_l1:,} children ({t_l1:.2f}s)")

        results.append({
            'm': m, 'corr': corr,
            'l0_surv': l0['n_survivors'],
            'l1_surv': len(surv_l1),
            'l1_children': ch_l1,
        })

    # Summary table
    print(f"\n  {'m':>5} {'corr':>10} {'L0_surv':>10} {'L1_children':>13} "
          f"{'L1_surv':>10}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*13} {'-'*10}")
    for r in results:
        print(f"  {r['m']:5d} {r['corr']:10.6f} {r['l0_surv']:10,} "
              f"{r['l1_children']:13,} {r['l1_surv']:10,}")

    return results


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Test pruning ideas')
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--c_target', type=float, default=1.28)
    parser.add_argument('--l2_sample', type=int, default=200,
                        help='Number of L1 parents to sample for L1->L2 test')
    args = parser.parse_args()

    m = args.m
    c_target = args.c_target
    n_half = 1
    d0 = 2

    print(f"Pruning Ideas Test")
    print(f"  m={m}, c_target={c_target}, d0={d0}")
    print(f"  correction={correction(m, n_half):.6f}")

    # ---- Run L0 ----
    print(f"\n{'#'*70}")
    print(f"PHASE 1: L0 -> L1  (d={d0} -> d={2*d0})")
    print(f"{'#'*70}")

    l0 = run_level0(n_half, m, c_target, verbose=True, d0=d0)
    l0_survivors = l0['survivors']
    print(f"L0 survivors: {len(l0_survivors):,}")

    if len(l0_survivors) == 0:
        print("PROVEN at L0 — nothing to test.")
        return

    n_half_child_l1 = d0  # at L1: n_half = n_half_0 * 2 = 1*2 = 2 = d0

    # Test Idea 1 on L0->L1
    test_idea1(l0_survivors, m, c_target, n_half_child_l1,
               label=f"[L0->L1, d={d0}->{2*d0}]")

    # Test Idea 2 on L0->L1
    l1_survivors = test_idea2(l0_survivors, m, c_target, n_half_child_l1,
                               label=f"[L0->L1, d={d0}->{2*d0}]")

    # ---- L1 -> L2 ----
    if len(l1_survivors) > 0:
        print(f"\n{'#'*70}")
        print(f"PHASE 2: L1 -> L2  (d={2*d0} -> d={4*d0})")
        print(f"{'#'*70}")

        n_half_child_l2 = 2 * d0  # n_half at L2 = n_half_0 * 4 = 4

        # Sample if too many parents
        n_sample = min(len(l1_survivors), args.l2_sample)
        if n_sample < len(l1_survivors):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(l1_survivors), n_sample, replace=False)
            l1_sample = l1_survivors[idx]
            print(f"Sampling {n_sample}/{len(l1_survivors)} L1 parents "
                  f"for L2 tests")
        else:
            l1_sample = l1_survivors
            print(f"Using all {len(l1_survivors)} L1 parents for L2 tests")

        # Test Idea 1 on L1->L2
        test_idea1(l1_sample, m, c_target, n_half_child_l2,
                   label=f"[L1->L2, d={2*d0}->{4*d0}]")

        # Test Idea 2 on L1->L2
        test_idea2(l1_sample, m, c_target, n_half_child_l2,
                   label=f"[L1->L2, d={2*d0}->{4*d0}]")

    # ---- Test Idea 3: varying m ----
    print(f"\n{'#'*70}")
    print(f"PHASE 3: Varying m (grid resolution)")
    print(f"{'#'*70}")
    test_idea3(c_target, m_values=(20, 30, 40, 50))

    print(f"\n{'='*70}")
    print(f"ALL TESTS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
