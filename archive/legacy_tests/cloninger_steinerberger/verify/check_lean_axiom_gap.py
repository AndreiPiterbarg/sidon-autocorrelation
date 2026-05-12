"""Check how many CPU-pruned compositions fail the Lean axiom threshold.

NOTE: This script was written under the old coarse-grid (S=m) parameterization.
The project has since switched to the C&S fine grid (S = 4nm), where the
threshold formula is:
  threshold = floor((c_target*m^2 + 3 + W_int/(2n) + eps) * 4n*ell)
The Lean axiom gap analysis below may need updating to match fine-grid thresholds.

The Lean axiom (FinalResult.lean line 79-84) requires:
  exists ell s_lo, test_value(n, m, c, ell, s_lo) > c_target + (4n/ell) * (1 + 2*W_int) / m^2

The CPU prunes with a different threshold:
  test_value > c_target + (3 + 2*W_int) / m^2

The CPU threshold is window-independent; the Lean threshold depends on ell.
At small ell, Lean is MUCH more demanding (factor 4n/ell >> 3).
At ell = 4n, Lean correction = (1+2W)/m^2 < CPU correction = (3+2W)/m^2.

Key question: for each CPU-pruned composition, does ANY window (ell, s_lo)
satisfy the Lean threshold? The CPU's killing window might not, but a
different window (larger ell) might work.
"""

import argparse
import math
import sys
import os
import time

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _check_lean_axiom(batch_int, n_half, m, c_target):
    """For each composition:
    1. Check if CPU prunes it (find killing window)
    2. Check if ANY window satisfies the Lean threshold

    Returns:
      cpu_pruned[b]: 1 if CPU prunes, 0 if survived
      lean_any_window[b]: 1 if exists (ell, s_lo) exceeding Lean threshold, 0 if not
      cpu_kill_ell[b]: ell at CPU killing window
      lean_best_ell[b]: ell at best Lean-exceeding window (smallest margin)
      lean_best_margin[b]: margin by which test_val exceeds Lean threshold (best window)
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1

    m_d = np.float64(m)
    m_sq = m_d * m_d
    n_d = np.float64(n_half)
    inv_4n = 1.0 / (4.0 * n_d)
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_sq
    d_minus_1 = d - 1

    max_ell = 2 * d
    cs_corr_base = c_target * m_sq + 3.0 + eps_margin
    ct_base_ell_arr = np.empty(max_ell + 1, dtype=np.float64)
    w_scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        ell_f = np.float64(ell)
        ct_base_ell_arr[ell] = cs_corr_base * ell_f * inv_4n
        w_scale_arr[ell] = 2.0 * ell_f * inv_4n

    cpu_pruned = np.zeros(B, dtype=np.int32)
    lean_any_window = np.zeros(B, dtype=np.int32)
    cpu_kill_ell = np.zeros(B, dtype=np.int32)
    lean_best_ell = np.zeros(B, dtype=np.int32)
    lean_best_margin = np.full(B, -1e30, dtype=np.float64)

    for b in prange(B):
        # Compute autoconvolution
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        # Prefix sum of bin masses
        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        # === Pass 1: CPU pruning (find killing window) ===
        cpu_killed = False
        for ell in range(2, max_ell + 1):
            if cpu_killed:
                break
            n_cv = ell - 1
            ct_base_ell = ct_base_ell_arr[ell]
            w_scale = w_scale_arr[ell]
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                dyn_x = ct_base_ell + w_scale * np.float64(W_int)
                dyn_it = np.int64(dyn_x * one_minus_4eps)
                if ws > dyn_it:
                    cpu_killed = True
                    cpu_pruned[b] = 1
                    cpu_kill_ell[b] = np.int32(ell)
                    break

        # === Pass 2: Lean threshold (check ALL windows) ===
        # Lean: test_value > c_target + (4n/ell) * (1 + 2*W_int) / m²
        # test_value = ws * 4n / (m² * ell)
        # So in integer space: ws > [c_target + (4n/ell)(1+2W)/m²] * m²ell/(4n)
        #                        = c_target*m²*ell/(4n) + (1+2W)
        # i.e., ws > c_target*m²*ell/(4n) + 1 + 2*W_int
        found_lean = False
        best_margin = -1e30
        best_ell = np.int32(0)
        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            ell_f = np.float64(ell)
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]

                # Lean threshold in integer space (old coarse-grid formula):
                # c_target*m^2*ell/(4n) + 1 + 2*W_int
                lean_int_thresh = c_target * m_sq * ell_f * inv_4n + 1.0 + 2.0 * np.float64(W_int)
                margin = np.float64(ws) - lean_int_thresh
                if margin > best_margin:
                    best_margin = margin
                    best_ell = np.int32(ell)
                if np.float64(ws) > lean_int_thresh:
                    found_lean = True

        if found_lean:
            lean_any_window[b] = 1
        lean_best_ell[b] = best_ell
        lean_best_margin[b] = best_margin

    return cpu_pruned, lean_any_window, cpu_kill_ell, lean_best_ell, lean_best_margin


def analyze_level(compositions, n_half, m, c_target, level_name=""):
    B, d = compositions.shape
    print(f"\n{'='*70}")
    print(f"Level {level_name}: {B:,} compositions, d={d}, n_half={n_half}")
    print(f"Parameters: m={m}, c_target={c_target}")
    four_n = 4 * n_half

    # Show threshold comparison
    print(f"  4n = {four_n}, max_ell = {2*d}")
    print(f"  CPU correction: (3 + 2W)/m² = (3 + 2W)/{m*m}")
    print(f"  Lean correction: ({four_n}/ell)(1 + 2W)/m²")
    print(f"  At ell=2:  Lean/CPU ratio (W=0) = {four_n/2:.0f}/3 = {four_n/(2*3):.1f}x")
    print(f"  At ell={four_n}: Lean/CPU ratio (W=0) = 1/3 = 0.33x (Lean easier)")
    print(f"{'='*70}")

    t0 = time.time()
    (cpu_pruned, lean_any, cpu_kill_ell, lean_best_ell, lean_best_margin
     ) = _check_lean_axiom(compositions.astype(np.int32), n_half, m, c_target)
    dt = time.time() - t0
    print(f"  Analysis time: {dt:.1f}s")

    n_cpu_pruned = cpu_pruned.sum()
    n_survived = B - n_cpu_pruned
    n_lean_ok = ((cpu_pruned == 1) & (lean_any == 1)).sum()
    n_lean_fail = ((cpu_pruned == 1) & (lean_any == 0)).sum()

    print(f"  CPU survived: {n_survived:,}  CPU pruned: {n_cpu_pruned:,}")
    print(f"\n  *** LEAN AXIOM CHECK (exists any window) ***")
    print(f"  CPU-pruned with exists Lean-valid window:  {n_lean_ok:,} ({100*n_lean_ok/max(1,n_cpu_pruned):.2f}%)")
    print(f"  CPU-pruned with NO Lean-valid window:  {n_lean_fail:,} ({100*n_lean_fail/max(1,n_cpu_pruned):.2f}%)")

    if n_lean_fail > 0:
        print(f"  >>> AXIOM FAILS for {n_lean_fail:,} compositions! <<<")
    else:
        print(f"  >>> AXIOM HOLDS for all CPU-pruned compositions <<<")

    # Distribution of CPU killing ells
    if n_cpu_pruned > 0:
        pruned_mask = cpu_pruned == 1
        cpu_ells = cpu_kill_ell[pruned_mask]
        lean_ells = lean_best_ell[pruned_mask]
        lean_margins = lean_best_margin[pruned_mask]
        lean_found = lean_any[pruned_mask]

        print(f"\n  CPU killing ell distribution:")
        for ell in range(2, 2*d + 1):
            count = (cpu_ells == ell).sum()
            if count > 0:
                # Of these, how many have a Lean-valid window?
                mask_ell = cpu_ells == ell
                lean_ok_here = (lean_found[mask_ell] == 1).sum()
                print(f"    ell={ell:3d}: {count:>8,} pruned, {lean_ok_here:>8,} Lean-ok ({100*lean_ok_here/count:.1f}%)")

    # Show some examples of failures
    if n_lean_fail > 0:
        fail_idx = np.where((cpu_pruned == 1) & (lean_any == 0))[0][:5]
        print(f"\n  Sample AXIOM FAILURES (closest margin):")
        for idx in fail_idx:
            ell_cpu = cpu_kill_ell[idx]
            best_ell = lean_best_ell[idx]
            margin = lean_best_margin[idx]
            comp = compositions[idx]
            # Compute the test value at best_ell for context
            print(f"    comp[{idx}]: CPU kills at ell={ell_cpu}, "
                  f"best Lean ell={best_ell}, margin={margin:.4f}")
            print(f"      composition: {comp[:min(16,d)]}{'...' if d>16 else ''}")

    # Also show: among survived compositions, how many have a Lean-valid window?
    # (These shouldn't — they survived CPU pruning)
    survived_with_lean = ((cpu_pruned == 0) & (lean_any == 1)).sum()
    if survived_with_lean > 0:
        print(f"\n  BONUS: {survived_with_lean:,} CPU-survivors have a Lean-valid window!")
        print(f"  (These could be pruned with a tighter CPU threshold)")

    return n_cpu_pruned, n_lean_ok, n_lean_fail


def main():
    parser = argparse.ArgumentParser(description="Check Lean axiom gap")
    parser.add_argument('--m', type=int, default=35)
    parser.add_argument('--n_half', type=int, default=2)
    parser.add_argument('--c_target', type=float, default=1.33)
    parser.add_argument('--max_level', type=int, default=3)
    parser.add_argument('--max_compositions', type=int, default=500_000)
    args = parser.parse_args()

    m = args.m
    n_half = args.n_half
    c_target = args.c_target

    print(f"Lean Axiom Gap Analysis: m={m}, n_half={n_half}, c_target={c_target}")
    print(f"m² = {m*m}")
    print()

    # Theoretical crossover by level
    print(f"Theoretical: Lean correction > CPU correction when ell < cutoff")
    print(f"  CPU correction (normalized): (3 + 2W) / {m*m}")
    print(f"  Lean correction (normalized): (4n/ell)(1 + 2W) / {m*m}")
    print()
    for level in range(args.max_level + 1):
        n_half_child = n_half * (2 ** level)
        d_child = 2 * n_half_child
        four_n = 4 * n_half_child
        # Lean > CPU when (4n/ell)(1+2W) > 3+2W -> ell < 4n(1+2W)/(3+2W)
        cutoff_W0 = four_n * 1.0 / 3.0
        cutoff_Wm = four_n * (1.0 + 2.0*m) / (3.0 + 2.0*m)
        print(f"  L{level}: d={d_child}, 4n={four_n}, "
              f"cutoff_W=0: ell<{cutoff_W0:.1f}, cutoff_W=m: ell<{cutoff_Wm:.1f}")

    # Generate/load compositions for each level
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    cs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(cs_dir, 'cloninger-steinerberger'))
    from compositions import generate_canonical_compositions_batched

    # L0
    d0 = 2 * n_half
    print(f"\n--- Generating L0 compositions: m={m} into d={d0} bins ---")
    batches = []
    total = 0
    for batch in generate_canonical_compositions_batched(d0, m):
        batches.append(batch)
        total += len(batch)
        if total > args.max_compositions:
            break
    comps = np.vstack(batches)[:args.max_compositions]
    print(f"  {len(comps):,} canonical compositions")
    analyze_level(comps, n_half, m, c_target, "L0")

    # Higher levels: generate children from checkpoint
    for level in range(1, args.max_level + 1):
        prev_ckpt = os.path.join(data_dir, f'checkpoint_L{level-1}_survivors.npy')
        if not os.path.exists(prev_ckpt):
            print(f"\n  No checkpoint at {prev_ckpt}, skipping L{level}")
            continue

        parents = np.load(prev_ckpt)
        d_parent = parents.shape[1]
        n_half_child = d_parent  # n_half_child = d_parent since d_child = 2*d_parent
        d_child = 2 * d_parent

        max_parents = min(20, len(parents))
        print(f"\n--- L{level}: generating children from {max_parents}/{len(parents):,} parents "
              f"(d_parent={d_parent} -> d_child={d_child}) ---")

        import itertools
        all_children = []
        for pi in range(max_parents):
            parent = parents[pi]
            ranges_list = [list(range(int(parent[k]) + 1)) for k in range(d_parent)]
            total_children = 1
            for r in ranges_list:
                total_children *= len(r)

            if total_children > 200_000:
                rng = np.random.default_rng(42 + pi)
                sample_size = min(50_000, total_children)
                children = np.empty((sample_size, d_child), dtype=np.int32)
                for si in range(sample_size):
                    for k in range(d_parent):
                        c = rng.integers(0, int(parent[k]) + 1)
                        children[si, 2*k] = c
                        children[si, 2*k + 1] = parent[k] - c
                all_children.append(children)
            else:
                children = np.empty((total_children, d_child), dtype=np.int32)
                for ci, combo in enumerate(itertools.product(*ranges_list)):
                    for k in range(d_parent):
                        children[ci, 2*k] = combo[k]
                        children[ci, 2*k + 1] = parent[k] - combo[k]
                all_children.append(children)

            if sum(len(c) for c in all_children) > args.max_compositions:
                break

        if all_children:
            all_children = np.vstack(all_children)[:args.max_compositions]
            # n_half for these children = d_parent // 2 * 2 = d_parent
            # Actually: n_half_child = d_child // 2 = d_parent
            analyze_level(all_children, d_parent, m, c_target, f"L{level} (d={d_child})")


if __name__ == '__main__':
    main()
