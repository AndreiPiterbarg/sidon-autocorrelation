"""Test L0 branch-and-bound correctness and performance.

Verifies that the B&B kernel produces identical survivors to the original
batch enumeration path, then benchmarks both on c_target=1.20.
"""
import sys
import os
import time
import numpy as np

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _project_dir)
sys.path.insert(0, os.path.join(_project_dir, 'cloninger-steinerberger'))

from cpu.run_cascade import run_level0


def sorted_rows(arr):
    """Sort array rows lexicographically for comparison."""
    if len(arr) == 0:
        return arr
    # Convert to list of tuples, sort, convert back
    return arr[np.lexsort(arr.T[::-1])]


def test_correctness(n_half, m, c_target, d0=None, use_flat=False):
    """Compare B&B and original L0 survivors — must be identical."""
    label = (f"n_half={n_half}, m={m}, c_target={c_target}"
             f"{f', d0={d0}' if d0 else ''}"
             f"{', flat' if use_flat else ''}")
    print(f"\n--- Correctness test: {label} ---")

    # Run original (batch) path
    t0 = time.time()
    orig = run_level0(n_half, m, c_target, verbose=False,
                      use_flat_threshold=use_flat, d0=d0, use_bnb=False)
    t_orig = time.time() - t0

    # Run B&B path
    t0 = time.time()
    bnb = run_level0(n_half, m, c_target, verbose=False,
                     use_flat_threshold=use_flat, d0=d0, use_bnb=True)
    t_bnb = time.time() - t0

    orig_surv = sorted_rows(orig['survivors'])
    bnb_surv = sorted_rows(bnb['survivors'])

    print(f"  Original: {orig['n_survivors']:,} survivors in {t_orig:.3f}s")
    print(f"  B&B:      {bnb['n_survivors']:,} survivors in {t_bnb:.3f}s")

    if orig['n_survivors'] != bnb['n_survivors']:
        print(f"  FAIL: survivor count mismatch "
              f"({orig['n_survivors']} vs {bnb['n_survivors']})")
        # Find differences
        if orig['n_survivors'] > 0 and bnb['n_survivors'] > 0:
            orig_set = set(map(tuple, orig_surv))
            bnb_set = set(map(tuple, bnb_surv))
            only_orig = orig_set - bnb_set
            only_bnb = bnb_set - orig_set
            if only_orig:
                print(f"    In original but not B&B ({len(only_orig)}):")
                for s in list(only_orig)[:5]:
                    print(f"      {s}")
            if only_bnb:
                print(f"    In B&B but not original ({len(only_bnb)}):")
                for s in list(only_bnb)[:5]:
                    print(f"      {s}")
        return False

    if orig['n_survivors'] > 0:
        if not np.array_equal(orig_surv, bnb_surv):
            print(f"  FAIL: survivor arrays differ despite same count")
            return False

    print(f"  PASS (speedup: {t_orig/t_bnb:.1f}x)" if t_bnb > 0
          else "  PASS")
    return True


def benchmark(n_half, m, c_target, d0=None, n_runs=3):
    """Benchmark B&B vs original on given parameters."""
    label = (f"n_half={n_half}, m={m}, c_target={c_target}"
             f"{f', d0={d0}' if d0 else ''}")
    print(f"\n=== Benchmark: {label} ===")

    # Warm up JIT
    print("  Warming up JIT...")
    run_level0(n_half, m, c_target, verbose=False, d0=d0, use_bnb=False)
    run_level0(n_half, m, c_target, verbose=False, d0=d0, use_bnb=True)

    orig_times = []
    bnb_times = []

    for i in range(n_runs):
        t0 = time.time()
        orig = run_level0(n_half, m, c_target, verbose=False, d0=d0,
                          use_bnb=False)
        orig_times.append(time.time() - t0)

        t0 = time.time()
        bnb = run_level0(n_half, m, c_target, verbose=False, d0=d0,
                         use_bnb=True)
        bnb_times.append(time.time() - t0)

    orig_med = np.median(orig_times)
    bnb_med = np.median(bnb_times)

    print(f"  Original: {orig_med:.4f}s (median of {n_runs})")
    print(f"  B&B:      {bnb_med:.4f}s (median of {n_runs})")
    print(f"  Speedup:  {orig_med/bnb_med:.1f}x")
    print(f"  Survivors: {orig['n_survivors']:,} (original) "
          f"vs {bnb['n_survivors']:,} (B&B)")
    return orig_med, bnb_med


if __name__ == '__main__':
    all_pass = True

    # --- Correctness tests ---
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    # Small cases
    all_pass &= test_correctness(1, 10, 1.20)
    all_pass &= test_correctness(1, 20, 1.20)
    all_pass &= test_correctness(2, 20, 1.20)
    all_pass &= test_correctness(2, 20, 1.30)

    # Odd d
    all_pass &= test_correctness(1, 20, 1.20, d0=3)

    # Flat threshold
    all_pass &= test_correctness(2, 20, 1.20, use_flat=True)
    all_pass &= test_correctness(2, 20, 1.30, use_flat=True)

    if not all_pass:
        print("\n*** SOME TESTS FAILED ***")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL CORRECTNESS TESTS PASSED")
    print("=" * 60)

    # --- Benchmark on c_target=1.20 ---
    print("\n" + "=" * 60)
    print("BENCHMARK (c_target=1.20)")
    print("=" * 60)

    benchmark(1, 20, 1.20)
    benchmark(2, 20, 1.20)
    benchmark(2, 20, 1.20, d0=3)

    # Larger m — B&B shines when composition space is large
    benchmark(1, 50, 1.20)
    benchmark(2, 50, 1.20)
    benchmark(2, 50, 1.20, d0=3)

    # Higher c_target — more aggressive pruning helps B&B
    print("\n" + "=" * 60)
    print("BENCHMARK (c_target=1.30)")
    print("=" * 60)
    benchmark(2, 20, 1.30)
    benchmark(2, 50, 1.30)
