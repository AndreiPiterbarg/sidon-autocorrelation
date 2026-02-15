"""Benchmark: final timing of optimized kernels."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))

import numpy as np
from pruning import correction
from test_values import compute_test_value_single
from solvers import (
    _find_min_eff_d4, _find_min_eff_d6,
    _prove_target_d4, _prove_target_d6,
    _build_interleaved_order,
    find_best_bound_direct, run_single_level,
)


def bench_d4(m, n_runs=3):
    n_half = 2
    d = 4
    S = 4 * n_half * m
    inv_m = 1.0 / m
    corr = correction(m)
    margin = 1.0 / (4.0 * m)
    a_uniform = np.full(d, float(4 * n_half) / d)
    init_tv = compute_test_value_single(a_uniform, n_half)
    init_min_eff = init_tv - corr
    c0_order = _build_interleaved_order(S // 2 + 1)

    _find_min_eff_d4(c0_order, S, n_half, inv_m, margin, corr, init_min_eff)  # warmup

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        thread_mins, thread_cfg = _find_min_eff_d4(c0_order, S, n_half, inv_m, margin, corr, init_min_eff)
        times.append(time.perf_counter() - t0)

    result = float(thread_mins[np.argmin(thread_mins)])
    best = min(times)
    med = sorted(times)[len(times) // 2]
    print(f"  d=4 m={m}: best={best:.3f}s  med={med:.3f}s  bound={result:.6f}")
    return best, result


def bench_d6(m, n_runs=3):
    n_half = 3
    d = 6
    S = 4 * n_half * m
    inv_m = 1.0 / m
    corr = correction(m)
    margin = 1.0 / (4.0 * m)
    a_uniform = np.full(d, float(4 * n_half) / d)
    init_tv = compute_test_value_single(a_uniform, n_half)
    init_min_eff = init_tv - corr
    c0_order = _build_interleaved_order(S // 2 + 1)

    _find_min_eff_d6(c0_order, S, n_half, inv_m, margin, corr, init_min_eff)  # warmup

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        thread_mins, thread_cfg = _find_min_eff_d6(c0_order, S, n_half, inv_m, margin, corr, init_min_eff)
        times.append(time.perf_counter() - t0)

    result = float(thread_mins[np.argmin(thread_mins)])
    best = min(times)
    med = sorted(times)[len(times) // 2]
    print(f"  d=6 m={m}: best={best:.3f}s  med={med:.3f}s  bound={result:.6f}")
    return best, result


def bench_prove(n_half, m, target, n_runs=3):
    S = 4 * n_half * m
    d = 2 * n_half
    inv_m = 1.0 / m
    corr = correction(m)
    margin = 1.0 / (4.0 * m)
    prune_target = target + corr
    fp_margin = 1e-9
    c0_order = _build_interleaved_order(S // 2 + 1)

    kernel = _prove_target_d4 if d == 4 else _prove_target_d6
    kernel(c0_order, S, n_half, inv_m, margin, prune_target, fp_margin)  # warmup

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        t_surv, t_asym, t_test, t_min_tv, t_min_cfg = kernel(
            c0_order, S, n_half, inv_m, margin, prune_target, fp_margin)
        times.append(time.perf_counter() - t0)

    n_survived = int(t_surv.sum())
    best = min(times)
    med = sorted(times)[len(times) // 2]
    print(f"  prove d={d} m={m} t={target}: best={best:.3f}s  med={med:.3f}s  surv={n_survived}")
    return best


if __name__ == '__main__':
    print("=" * 60)
    print("OPTIMIZED BENCHMARK (interleaved d=4, flat triples d=6)")
    print("=" * 60)

    print("\n--- find_min_eff d=4 ---")
    bench_d4(m=100)
    bench_d4(m=200)

    print("\n--- find_min_eff d=6 ---")
    bench_d6(m=8)
    bench_d6(m=10)
    bench_d6(m=12)

    print("\n--- prove_target d=4 ---")
    bench_prove(2, 100, 0.9)
    bench_prove(2, 200, 0.9)

    print("\n--- prove_target d=6 ---")
    bench_prove(3, 8, 0.7)
    bench_prove(3, 10, 0.7)
