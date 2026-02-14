"""Benchmark: find_best_bound_direct vs find_best_bound (binary search)."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))

from core import (
    find_best_bound, find_best_bound_direct,
    correction, count_compositions,
)


def bench(func, n_half, m, repeats=3, **kwargs):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = func(n_half=n_half, m=m, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, min(times)


def main():
    configs = [
        (2, 10),
        (2, 20),
        (2, 50),
        (3, 3),
        (3, 5),
    ]

    print(f"{'Config':<30} {'binary_search':>15} {'direct':>15} {'speedup':>10}")
    print("-" * 75)

    for n_half, m in configs:
        d = 2 * n_half
        S = 4 * n_half * m
        n_total = count_compositions(d, S)

        repeats = 3 if n_total < 200000 else 1

        bs_result, bs_time = bench(
            find_best_bound, n_half, m, repeats=repeats,
            lo=0.8, hi=1.5, tol=0.005, verbose=False)

        direct_result, direct_time = bench(
            find_best_bound_direct, n_half, m, repeats=repeats,
            verbose=False)

        speedup = bs_time / direct_time if direct_time > 0 else 0

        bs_str = f"{bs_result:.6f}" if bs_result is not None else "None"
        dir_str = f"{direct_result:.6f}"

        label = f"n={n_half}, m={m} ({n_total:,} comps)"
        print(f"  {label:<28} {bs_str:>8} {bs_time:5.2f}s  "
              f"{dir_str:>8} {direct_time:5.2f}s  {speedup:6.1f}x")


if __name__ == "__main__":
    main()
