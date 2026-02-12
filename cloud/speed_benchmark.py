"""
speed_benchmark.py — Benchmark harness for Sidon autocorrelation optimization.

Times key computational bottlenecks individually and runs standardized
optimization benchmarks at small and medium P values.
"""

import time
import json
import numpy as np
from sidon_core import (
    autoconv_coeffs, lse_obj_nb, lse_grad_nb, armijo_step_nb,
    _polyak_polish_nb, _hybrid_single_restart, warmup,
    BETA_HEAVY,
)


def bench_autoconv(P, n_calls=10000):
    """Benchmark autoconv_coeffs at a given P."""
    x = np.random.dirichlet(np.ones(P))
    # Warm call
    _ = autoconv_coeffs(x, P)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        c = autoconv_coeffs(x, P)
    dt = time.perf_counter() - t0
    return dt, n_calls


def bench_lse_grad(P, n_calls=5000):
    """Benchmark lse_grad_nb at a given P."""
    x = np.random.dirichlet(np.ones(P))
    beta = 100.0
    _ = lse_grad_nb(x, P, beta)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        g = lse_grad_nb(x, P, beta)
    dt = time.perf_counter() - t0
    return dt, n_calls


def bench_polyak(P, n_iters=50000):
    """Benchmark Polyak polish."""
    x = np.random.dirichlet(np.ones(P))
    _ = _polyak_polish_nb(x, P, 10)  # warm
    t0 = time.perf_counter()
    val, xopt = _polyak_polish_nb(x, P, n_iters)
    dt = time.perf_counter() - t0
    return dt, n_iters, float(val)


def bench_full_restart(P, n_iters_lse=1000, n_iters_polyak=50000):
    """Benchmark a single full hybrid restart."""
    x = np.random.dirichlet(np.ones(P))
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    # Warm
    _ = _hybrid_single_restart(np.ones(5) / 5.0, 5,
                                np.array([1.0, 10.0]), 10, 10)
    t0 = time.perf_counter()
    lse_v, pol_v, pol_x = _hybrid_single_restart(x, P, beta_arr,
                                                   n_iters_lse, n_iters_polyak)
    dt = time.perf_counter() - t0
    return dt, float(lse_v), float(pol_v)


def run_benchmark():
    print("=" * 70)
    print("SIDON AUTOCORRELATION SPEED BENCHMARK")
    print("=" * 70)

    print("\nWarming up Numba JIT...")
    t0 = time.perf_counter()
    warmup()
    warmup_time = time.perf_counter() - t0
    print(f"  Warmup: {warmup_time:.2f}s")

    np.random.seed(42)
    results = {}

    # ── Autoconvolution benchmarks ──
    print("\n--- Autoconvolution (autoconv_coeffs) ---")
    for P in [30, 50, 100, 200]:
        n_calls = max(1000, 50000 // P)
        dt, nc = bench_autoconv(P, n_calls)
        us_per_call = dt / nc * 1e6
        results[f'autoconv_P{P}'] = {
            'time_s': dt, 'calls': nc, 'us_per_call': us_per_call
        }
        print(f"  P={P:>4}: {us_per_call:8.2f} us/call  ({nc} calls in {dt:.3f}s)")

    # ── Gradient benchmarks ──
    print("\n--- LSE Gradient (lse_grad_nb) ---")
    for P in [30, 50, 100, 200]:
        n_calls = max(500, 20000 // P)
        dt, nc = bench_lse_grad(P, n_calls)
        us_per_call = dt / nc * 1e6
        results[f'lse_grad_P{P}'] = {
            'time_s': dt, 'calls': nc, 'us_per_call': us_per_call
        }
        print(f"  P={P:>4}: {us_per_call:8.2f} us/call  ({nc} calls in {dt:.3f}s)")

    # ── Polyak polish benchmarks ──
    print("\n--- Polyak Polish (_polyak_polish_nb) ---")
    for P, n_iters in [(30, 50000), (50, 50000), (100, 20000)]:
        dt, ni, val = bench_polyak(P, n_iters)
        iters_per_sec = ni / dt
        results[f'polyak_P{P}_{n_iters // 1000}k'] = {
            'time_s': dt, 'iters': ni, 'iters_per_sec': iters_per_sec,
            'final_val': val
        }
        print(f"  P={P:>4}, {ni//1000}k iters: {dt:7.3f}s  "
              f"({iters_per_sec:,.0f} iters/s)  val={val:.6f}")

    # ── Full restart benchmarks ──
    print("\n--- Full Hybrid Restart (_hybrid_single_restart) ---")
    for P, n_lse, n_pol in [(30, 1000, 50000), (50, 500, 30000), (100, 200, 10000)]:
        dt, lse_v, pol_v = bench_full_restart(P, n_lse, n_pol)
        results[f'full_restart_P{P}'] = {
            'time_s': dt, 'n_iters_lse': n_lse, 'n_iters_polyak': n_pol,
            'lse_val': lse_v, 'pol_val': pol_v
        }
        print(f"  P={P:>4} (lse={n_lse}, pol={n_pol//1000}k): "
              f"{dt:7.3f}s  lse={lse_v:.6f}  pol={pol_v:.6f}")

    # ── Production-scale single restart (the real bottleneck) ──
    print("\n--- Production-scale single restart ---")
    P = 50
    n_lse = 10000
    n_pol = 200000
    x = np.random.dirichlet(np.ones(P))
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    t0 = time.perf_counter()
    lse_v, pol_v, pol_x = _hybrid_single_restart(x, P, beta_arr, n_lse, n_pol)
    dt = time.perf_counter() - t0
    results['production_P50'] = {
        'time_s': dt, 'n_iters_lse': n_lse, 'n_iters_polyak': n_pol,
        'lse_val': float(lse_v), 'pol_val': float(pol_v)
    }
    print(f"  P=50 (lse=10000, pol=200k): {dt:.3f}s  "
          f"lse={lse_v:.6f}  pol={pol_v:.6f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_time = sum(v['time_s'] for v in results.values())
    print(f"  Total benchmark time: {total_time:.2f}s")
    print(f"  Production P=50 single restart: {results['production_P50']['time_s']:.3f}s")

    # Save baseline
    with open('benchmark_baseline.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved to benchmark_baseline.json")

    return results


if __name__ == '__main__':
    run_benchmark()
