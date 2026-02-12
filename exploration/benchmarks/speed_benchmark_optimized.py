"""
Optimized speed benchmark — run AFTER baseline to compare.

Tests the optimized methods from optimized_methods.py using fast_core.py.
Also benchmarks the individual optimized components.
"""

import sys, os, time, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")

from sidon_core import (
    autoconv_coeffs, project_simplex_nb,
    _polyak_polish_nb, _hybrid_single_restart, lse_objgrad_nb,
    BETA_HEAVY, BETA_ULTRA
)
from experiments.speed_tests.fast_core import (
    autoconv_single_k,
    _polyak_polish_fast, _hybrid_single_restart_fast, warmup as warmup_fast
)
from experiments.speed_tests.optimized_methods import ALL_METHODS_FAST

P = 200
TIME_BUDGET = 30


def verify_solution(x):
    c = autoconv_coeffs(x, P)
    return float(np.max(c))


# ============================================================================
# Component benchmarks (optimized vs original)
# ============================================================================

def bench_polyak_comparison(n_iters=50000):
    """Compare original vs fast Polyak polish."""
    rng = np.random.default_rng(42)
    x = rng.dirichlet(np.ones(P))

    # Original
    x_orig = x.copy()
    t0 = time.perf_counter()
    val_orig, x_out_orig = _polyak_polish_nb(x_orig, P, n_iters)
    t_orig = time.perf_counter() - t0

    # Fast
    x_fast = x.copy()
    t0 = time.perf_counter()
    val_fast, x_out_fast = _polyak_polish_fast(x_fast, P, n_iters)
    t_fast = time.perf_counter() - t0

    return {
        "original_s": t_orig,
        "fast_s": t_fast,
        "speedup": t_orig / t_fast if t_fast > 0 else 0,
        "original_val": val_orig,
        "fast_val": val_fast,
        "val_diff": val_fast - val_orig,
    }


def bench_hybrid_comparison(n_iters_lse=500, n_iters_polyak=10000):
    """Compare original vs fast hybrid single restart."""
    rng = np.random.default_rng(42)
    x = rng.dirichlet(np.ones(P))
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    # Original
    x_orig = x.copy()
    t0 = time.perf_counter()
    lse_o, pol_o, x_o = _hybrid_single_restart(x_orig, P, beta_arr, n_iters_lse, n_iters_polyak)
    t_orig = time.perf_counter() - t0

    # Fast
    x_fast = x.copy()
    t0 = time.perf_counter()
    lse_f, pol_f, x_f = _hybrid_single_restart_fast(x_fast, P, beta_arr, n_iters_lse, n_iters_polyak)
    t_fast = time.perf_counter() - t0

    return {
        "original_s": t_orig,
        "fast_s": t_fast,
        "speedup": t_orig / t_fast if t_fast > 0 else 0,
        "original_val": pol_o,
        "fast_val": pol_f,
        "val_diff": pol_f - pol_o,
    }


def bench_single_k(n_calls=50000):
    """Benchmark autoconv_single_k vs full autoconv_coeffs."""
    rng = np.random.default_rng(42)
    x = rng.dirichlet(np.ones(P))

    t0 = time.perf_counter()
    for _ in range(n_calls):
        c = autoconv_coeffs(x, P)
    t_full = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(n_calls):
        v = autoconv_single_k(x, P, P - 1)
    t_single = (time.perf_counter() - t0) / n_calls

    return {
        "full_autoconv_us": t_full * 1e6,
        "single_k_us": t_single * 1e6,
        "speedup": t_full / t_single if t_single > 0 else 0,
    }


# ============================================================================
# Method benchmarks
# ============================================================================

def run_method_benchmarks():
    """Run each optimized method for TIME_BUDGET seconds."""
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZED METHOD BENCHMARKS (P={P}, budget={TIME_BUDGET}s each)")
    print(f"{'=' * 70}")

    results = {}
    for key, (name, func) in ALL_METHODS_FAST.items():
        print(f"\n  Running {name}...", end=" ", flush=True)
        val, x, restarts, elapsed = func(TIME_BUDGET, seed=42)
        results[key] = {
            "name": name,
            "value": val,
            "restarts": restarts,
            "elapsed": elapsed,
            "restarts_per_s": restarts / elapsed if elapsed > 0 else 0,
        }
        print(f"val={val:.6f}  restarts={restarts}  "
              f"time={elapsed:.1f}s  rate={restarts/elapsed:.2f} restarts/s")

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Warmup original
    print("Warming up Numba (original)...")
    x_test = np.ones(5) / 5.0
    _ = autoconv_coeffs(x_test, 5)
    _ = _polyak_polish_nb(x_test, 5, 10)
    beta_arr = np.array([1.0, 10.0])
    _ = _hybrid_single_restart(x_test, 5, beta_arr, 10, 10)
    _ = lse_objgrad_nb(x_test, 5, 10.0)
    _ = project_simplex_nb(x_test)

    # Warmup fast
    print("Warming up Numba (fast)...")
    warmup_fast()
    print("Warmup complete.\n")

    # Component comparisons
    print("=" * 70)
    print("COMPONENT COMPARISONS")
    print("=" * 70)

    sk_results = bench_single_k(50000)
    print(f"\n  autoconv_coeffs (full): {sk_results['full_autoconv_us']:.1f} us")
    print(f"  autoconv_single_k:      {sk_results['single_k_us']:.1f} us")
    print(f"  Speedup (single k):     {sk_results['speedup']:.1f}x")

    polyak_results = bench_polyak_comparison(50000)
    print(f"\n  Polyak 50K original:  {polyak_results['original_s']:.2f}s (val={polyak_results['original_val']:.6f})")
    print(f"  Polyak 50K fast:      {polyak_results['fast_s']:.2f}s (val={polyak_results['fast_val']:.6f})")
    print(f"  Polyak speedup:       {polyak_results['speedup']:.2f}x")
    print(f"  Polyak val diff:      {polyak_results['val_diff']:+.8f}")

    hybrid_results = bench_hybrid_comparison(500, 10000)
    print(f"\n  Hybrid (500+10K) original: {hybrid_results['original_s']:.2f}s (val={hybrid_results['original_val']:.6f})")
    print(f"  Hybrid (500+10K) fast:     {hybrid_results['fast_s']:.2f}s (val={hybrid_results['fast_val']:.6f})")
    print(f"  Hybrid speedup:            {hybrid_results['speedup']:.2f}x")
    print(f"  Hybrid val diff:           {hybrid_results['val_diff']:+.8f}")

    # Method benchmarks
    method_results = run_method_benchmarks()

    # Load baseline for comparison
    baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "benchmark_baseline.json")
    baseline = None
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)

    # Summary
    print(f"\n\n{'=' * 70}")
    print("COMPARISON: BASELINE vs OPTIMIZED")
    print(f"{'=' * 70}")
    print(f"{'Method':<25s} {'Base Rate':>10s} {'Fast Rate':>10s} {'Speedup':>8s} "
          f"{'Base Val':>10s} {'Fast Val':>10s} {'Quality':>8s}")
    print(f"{'-' * 25} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}")

    for key in ALL_METHODS_FAST:
        fast_r = method_results[key]
        if baseline and key in baseline.get("methods", {}):
            base_r = baseline["methods"][key]
            base_rate = base_r["restarts_per_s"]
            base_val = base_r["value"]
            speedup = fast_r["restarts_per_s"] / base_rate if base_rate > 0 else 0
            quality = "OK" if abs(fast_r["value"] - base_val) / base_val < 0.01 else "WARN"
        else:
            base_rate = 0
            base_val = 0
            speedup = 0
            quality = "N/A"

        name_short = key[:25]
        print(f"{name_short:<25s} {base_rate:8.3f}/s {fast_r['restarts_per_s']:8.3f}/s "
              f"{speedup:7.2f}x {base_val:10.6f} {fast_r['value']:10.6f} {quality:>8s}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_optimized.json")
    save_data = {
        "P": P,
        "time_budget": TIME_BUDGET,
        "components": {
            "single_k": sk_results,
            "polyak": polyak_results,
            "hybrid": hybrid_results,
        },
        "methods": method_results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_path}")
