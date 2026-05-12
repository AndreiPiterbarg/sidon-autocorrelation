"""Benchmark: MATLAB-style vs Python Gray-code cascade at m=50, n_half=3, c_target=1.28.

Measures parents/second for both approaches at L1:
  1. MATLAB-style: numpy-vectorized batch pruning (analogous to MATLAB's matmul approach)
  2. Python cascade: Numba JIT Gray-code fused kernel
"""

import sys
import os
import time
import math
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _repo_dir)
sys.path.insert(0, os.path.join(_repo_dir, 'cloninger-steinerberger'))

os.environ['NUMBA_DISABLE_JIT'] = '0'

from cpu.run_cascade import (run_level0, process_parent_fused,
                              _compute_bin_ranges, _fused_generate_and_prune_gray,
                              _tighten_ranges, _default_buf_cap)
from pruning import correction


# ===================================================================
# MATLAB-style pruning (vectorized numpy, faithful to original_baseline_matlab.m)
# ===================================================================

def matlab_process_one_parent(parent_int, d_child, n_half, m, c_target):
    """Process one parent MATLAB-style: generate all children, prune via vectorized ops.

    Faithfully reimplements the MATLAB algorithm from original_baseline_matlab.m:
    1. Split parent bins into child bins with cursor ranges
    2. Enumerate ALL children (Cartesian product)
    3. Compute autoconvolution as vectorized pairwise products
    4. Window scan with MATLAB threshold: (c_target + 1/m^2) + 2/m * W

    Returns: (n_children_generated, n_survivors, elapsed_seconds)
    """
    d_parent = len(parent_int)
    S = 4 * n_half * m
    gridSpace = 1.0 / m

    # Convert parent to continuous weights
    parent_cont = parent_int.astype(np.float64) / S

    # x = sqrt(lowerBound / numBins) — max single-bin mass (MATLAB line 138)
    x = math.sqrt(c_target / d_child)

    # Step 1: compute cursor ranges (MATLAB lines 142-153)
    ranges = []
    for j in range(d_parent):
        weight = parent_cont[j]
        start = round((weight - x) / gridSpace) * gridSpace
        end_point = round(min(weight, x) / gridSpace) * gridSpace
        sub_bins = np.arange(max(0, start), end_point + gridSpace / 2, gridSpace)
        sub_bins = sub_bins[sub_bins >= -1e-12]
        sub_bins = sub_bins[sub_bins <= weight + 1e-12]
        if len(sub_bins) == 0:
            sub_bins = np.array([max(0, min(weight, x))])
        ranges.append(sub_bins)

    cart_size = 1
    for r in ranges:
        cart_size *= len(r)

    if cart_size == 0:
        return 0, 0, 0.0

    t0 = time.perf_counter()

    # Step 2: enumerate children (MATLAB lines 177-189)
    MAX_BATCH = 2_000_000
    if cart_size > MAX_BATCH:
        # Skip extremely large parents for timing fairness
        return cart_size, -1, 0.0

    grids = np.meshgrid(*ranges, indexing='ij')
    flat = [g.ravel() for g in grids]
    actual_count = len(flat[0])
    children = np.empty((actual_count, d_child), dtype=np.float64)
    for i in range(d_parent):
        children[:, 2 * i] = flat[i]
        children[:, 2 * i + 1] = parent_cont[i] - flat[i]

    # Step 3: Autoconvolution (MATLAB lines 194-196)
    n_conv = 2 * d_child - 1
    conv_all = np.zeros((actual_count, n_conv), dtype=np.float64)
    for i in range(d_child):
        for j in range(d_child):
            conv_all[:, i + j] += children[:, i] * children[:, j]

    # Step 4: Window scan with MATLAB threshold (MATLAB lines 207-233)
    survived = np.ones(actual_count, dtype=bool)

    for ell in range(2, 2 * d_child + 1):
        if not np.any(survived):
            break

        n_cv = ell - 1
        n_windows = n_conv - n_cv + 1
        idx = np.where(survived)[0]
        if len(idx) == 0:
            break

        sub_conv = conv_all[idx]
        sub_children = children[idx]

        for s in range(n_windows):
            ws = np.sum(sub_conv[:, s:s + n_cv], axis=1)
            TV = ws * (2 * d_child) / ell

            lo_bin = max(0, s - (d_child - 1))
            hi_bin = min(d_child - 1, s + n_cv - 1)
            W = np.sum(sub_children[:, lo_bin:hi_bin + 1], axis=1)

            bound = (c_target + gridSpace ** 2) + 2 * gridSpace * W
            pruned = TV >= bound

            if np.any(pruned):
                survived[idx[pruned]] = False
                # Update idx for next window
                idx = np.where(survived)[0]
                if len(idx) == 0:
                    break
                sub_conv = conv_all[idx]
                sub_children = children[idx]

    elapsed = time.perf_counter() - t0
    n_survived = int(np.sum(survived))
    return actual_count, n_survived, elapsed


# ===================================================================
# Main benchmark
# ===================================================================

def main():
    n_half = 3
    m = 50
    c_target = 1.28

    d0 = 2 * n_half   # = 6
    d_child = 2 * d0  # = 12
    n_half_child = d_child // 2  # = 6
    S = 4 * n_half * m  # = 600

    print("=" * 70)
    print(f"  BENCHMARK: MATLAB vs Python")
    print(f"  m={m}, n_half={n_half}, c_target={c_target}")
    print(f"  d_parent={d0}, d_child={d_child}, S={S}, gridSpace={1/m}")
    print("=" * 70)

    # --- L0: generate survivors (shared starting point) ---
    print("\n--- L0: Generating survivors ---")
    t0 = time.perf_counter()
    l0 = run_level0(n_half, m, c_target, verbose=True)
    t_l0 = time.perf_counter() - t0

    survivors = l0['survivors']
    n_total_comps = l0['n_processed']
    n_survivors = l0['n_survivors']

    print(f"\n  L0: {n_total_comps:,} compositions -> {n_survivors:,} survivors in {t_l0:.2f}s")
    print(f"  L0 throughput: {n_total_comps / t_l0:,.0f} compositions/s")

    if n_survivors == 0:
        print("  No survivors at L0 — proven! Nothing to benchmark at L1.")
        return

    # --- Select parents for L1 benchmark ---
    N_BENCH = min(50, n_survivors)  # benchmark on 50 parents
    bench_parents = survivors[:N_BENCH]
    print(f"\n  Using {N_BENCH} parents for L1 benchmark")

    # --- Warmup: JIT compile the Numba kernel ---
    print("\n--- Warming up Numba JIT ---")
    t_warm = time.perf_counter()
    _ = process_parent_fused(bench_parents[0], m, c_target, n_half_child)
    print(f"  Warmup done in {time.perf_counter() - t_warm:.2f}s")

    # --- Python Gray-code cascade L1 ---
    print("\n--- Python Gray-code L1 ---")
    py_total_children = 0
    py_total_survived = 0
    t0 = time.perf_counter()

    for pi in range(N_BENCH):
        surv, n_ch = process_parent_fused(bench_parents[pi], m, c_target, n_half_child)
        py_total_children += n_ch
        py_total_survived += len(surv)

    t_py = time.perf_counter() - t0
    py_rate = N_BENCH / t_py if t_py > 0 else 0

    print(f"  Python: {N_BENCH} parents in {t_py:.2f}s")
    print(f"  Python: {py_rate:.2f} parents/s")
    print(f"  Python: {py_total_children:,} total children tested")
    print(f"  Python: {py_total_survived:,} survivors")
    print(f"  Python: avg {py_total_children / N_BENCH:,.0f} children/parent")
    if t_py > 0:
        print(f"  Python: {py_total_children / t_py:,.0f} children/s")

    # --- MATLAB-style L1 ---
    print("\n--- MATLAB-style (numpy vectorized) L1 ---")
    mat_total_children = 0
    mat_total_survived = 0
    mat_skipped = 0
    t0 = time.perf_counter()

    for pi in range(N_BENCH):
        n_gen, n_surv, _ = matlab_process_one_parent(
            bench_parents[pi], d_child, n_half, m, c_target)
        if n_surv == -1:
            mat_skipped += 1
            continue
        mat_total_children += n_gen
        mat_total_survived += n_surv

    t_mat = time.perf_counter() - t0
    mat_processed = N_BENCH - mat_skipped
    mat_rate = mat_processed / t_mat if t_mat > 0 else 0

    print(f"  MATLAB: {mat_processed} parents in {t_mat:.2f}s (skipped {mat_skipped} too large)")
    print(f"  MATLAB: {mat_rate:.2f} parents/s")
    print(f"  MATLAB: {mat_total_children:,} total children tested")
    print(f"  MATLAB: {mat_total_survived:,} survivors")
    if mat_processed > 0:
        print(f"  MATLAB: avg {mat_total_children / mat_processed:,.0f} children/parent")
    if t_mat > 0:
        print(f"  MATLAB: {mat_total_children / t_mat:,.0f} children/s")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Parameters: m={m}, n_half={n_half}, c_target={c_target}")
    print(f"  d_parent={d0}, d_child={d_child}")
    print(f"\n  {'Metric':<35} {'Python':>15} {'MATLAB-numpy':>15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")
    print(f"  {'Parents/second':<35} {py_rate:>15.2f} {mat_rate:>15.2f}")
    print(f"  {'Total children tested':<35} {py_total_children:>15,} {mat_total_children:>15,}")
    print(f"  {'Total survivors':<35} {py_total_survived:>15,} {mat_total_survived:>15,}")
    print(f"  {'Wall time (s)':<35} {t_py:>15.2f} {t_mat:>15.2f}")

    if mat_rate > 0 and py_rate > 0:
        speedup = py_rate / mat_rate
        print(f"\n  Python speedup over MATLAB-numpy: {speedup:.1f}x")

    print(f"\n  IMPORTANT CAVEATS:")
    print(f"  - MATLAB-style here uses numpy on CPU, NOT actual MATLAB with GPU arrays")
    print(f"  - Original MATLAB used 3 GPU workers (gpuArray + spmd)")
    print(f"  - For d_child={d_child}, the MATLAB matrix ops are O(d^2 * n_children)")
    print(f"  - MATLAB GPU would be faster than numpy for large batches")
    print(f"  - Python Gray-code is O(d * n_children) with incremental updates")
    print(f"  - The algorithmic difference (matrix vs incremental) is the key insight")


if __name__ == '__main__':
    main()
