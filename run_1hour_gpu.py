"""1-hour execution: find the best provable lower bound on C_{1a} — GPU version.

Same strategy as run_1hour.py but using CUDA GPU kernels instead of CPU/Numba.
Compares directly against the CPU baseline to measure GPU speedup.

Strategy:
  - d=4 with large m: raw minimum ~1.09, small correction -> good bounds
  - d=6 with moderate m: raw minimum higher, larger correction
  - d=8 with small-moderate m: even higher raw minimum
  - Run progressively larger configs, tracking the best bound
"""
import sys
import os
import time
import json
from math import comb
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
from pruning import correction, count_compositions
from gpu import gpu_find_best_bound_direct, is_available, get_device_name


TOTAL_BUDGET = 3600  # 1 hour in seconds
RESERVE = 60         # reserve for warmup + I/O


def fmt_count(n):
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    else:
        return f"{n:,}"


def run_config(n_half, m, results, deadline):
    """Run gpu_find_best_bound_direct for one (n_half, m) config.

    Returns the bound, or None if skipped due to time.
    """
    d = 2 * n_half
    S = 4 * n_half * m
    n_configs = count_compositions(d, S)
    corr = correction(m)
    time_left = deadline - time.time()

    # Estimate time from previous runs at same d
    prev_rates = [r['rate'] for r in results if r['d'] == d and r.get('rate')]
    if prev_rates:
        est_rate = sum(prev_rates) / len(prev_rates)
        est_time = n_configs / est_rate
    else:
        # Conservative initial estimates for GPU (configs/sec)
        # GPU should be much faster than CPU, but start conservative
        est_rate = {4: 5e9, 6: 2e9, 8: 1e9, 10: 5e8, 12: 3e8}.get(d, 1e8)
        est_time = n_configs / est_rate

    if est_time > time_left:
        print(f"  SKIP n={n_half} m={m}: ~{fmt_count(n_configs)} configs, "
              f"est {est_time:.0f}s > {time_left:.0f}s remaining")
        return None

    print(f"\n{'='*70}")
    print(f"  n_half={n_half}, m={m} (d={d}, S={S})")
    print(f"  Compositions: {fmt_count(n_configs)}  |  Correction: {corr:.6f}")
    print(f"  Time remaining: {time_left:.0f}s  |  Estimated: {est_time:.0f}s")
    print(f"{'='*70}")

    t0 = time.time()
    bound = gpu_find_best_bound_direct(n_half, m, verbose=True)
    elapsed = time.time() - t0
    rate = n_configs / max(elapsed, 0.001)

    entry = {
        'n_half': n_half, 'm': m, 'd': d, 'S': S,
        'n_configs': n_configs, 'correction': corr,
        'bound': bound, 'elapsed': elapsed, 'rate': rate,
    }
    results.append(entry)

    print(f"  Time: {elapsed:.1f}s  |  Rate: {fmt_count(int(rate))}/s")
    return bound


def main():
    # Check GPU availability first
    if not is_available():
        print("ERROR: CUDA GPU not available. Cannot run GPU version.")
        print("Make sure CUDA drivers are installed and a GPU is present.")
        sys.exit(1)

    device = get_device_name()

    start = time.time()
    deadline = start + TOTAL_BUDGET - RESERVE

    print("=" * 70)
    print("  1-HOUR BOUND SEARCH: GPU Version (Cloninger-Steinerberger)")
    print(f"  GPU Device: {device}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Budget: {TOTAL_BUDGET}s ({TOTAL_BUDGET/60:.0f} min)")
    print("=" * 70)

    results = []
    best_bound = -1.0
    best_config = None

    # =====================================================================
    # Phase 1: GPU warmup (compile/initialize)
    # =====================================================================
    print("\n--- Phase 1: GPU warmup ---")
    for n_half, m in [(2, 10), (3, 3)]:
        b = run_config(n_half, m, results, deadline)
        if b is not None and b > best_bound:
            best_bound = b
            best_config = (n_half, m)

    # =====================================================================
    # Phase 2: d=4 with increasing m
    # =====================================================================
    print("\n\n--- Phase 2: d=4 (n_half=2) scaling m ---")
    d4_ms = [50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000]
    for m in d4_ms:
        if time.time() >= deadline:
            print("  Time budget exhausted.")
            break
        b = run_config(2, m, results, deadline)
        if b is not None and b > best_bound:
            best_bound = b
            best_config = (2, m)
            print(f"  *** NEW BEST: {best_bound:.6f} (n=2, m={m}) ***")

    # =====================================================================
    # Phase 3: d=6 with increasing m
    # =====================================================================
    print("\n\n--- Phase 3: d=6 (n_half=3) scaling m ---")
    d6_ms = [5, 10, 16, 20, 25, 30, 40, 50, 60, 80, 100]
    for m in d6_ms:
        if time.time() >= deadline:
            print("  Time budget exhausted.")
            break
        b = run_config(3, m, results, deadline)
        if b is not None and b > best_bound:
            best_bound = b
            best_config = (3, m)
            print(f"  *** NEW BEST: {best_bound:.6f} (n=3, m={m}) ***")

    # =====================================================================
    # Phase 4: d=8 with increasing m
    # =====================================================================
    time_left = deadline - time.time()
    if time_left > 30:
        print(f"\n\n--- Phase 4: d=8 (n_half=4), {time_left:.0f}s remaining ---")
        d8_ms = [3, 5, 8, 10, 12, 15, 20, 25]
        for m in d8_ms:
            if time.time() >= deadline:
                print("  Time budget exhausted.")
                break
            b = run_config(4, m, results, deadline)
            if b is not None and b > best_bound:
                best_bound = b
                best_config = (4, m)
                print(f"  *** NEW BEST: {best_bound:.6f} (n=4, m={m}) ***")

    # =====================================================================
    # Phase 5: d=10 if time permits
    # =====================================================================
    time_left = deadline - time.time()
    if time_left > 30:
        print(f"\n\n--- Phase 5: d=10 (n_half=5), {time_left:.0f}s remaining ---")
        d10_ms = [3, 5, 8, 10]
        for m in d10_ms:
            if time.time() >= deadline:
                print("  Time budget exhausted.")
                break
            b = run_config(5, m, results, deadline)
            if b is not None and b > best_bound:
                best_bound = b
                best_config = (5, m)
                print(f"  *** NEW BEST: {best_bound:.6f} (n=5, m={m}) ***")

    # =====================================================================
    # Phase 6: d=12 if time permits
    # =====================================================================
    time_left = deadline - time.time()
    if time_left > 30:
        print(f"\n\n--- Phase 6: d=12 (n_half=6), {time_left:.0f}s remaining ---")
        d12_ms = [3, 5, 8]
        for m in d12_ms:
            if time.time() >= deadline:
                print("  Time budget exhausted.")
                break
            b = run_config(6, m, results, deadline)
            if b is not None and b > best_bound:
                best_bound = b
                best_config = (6, m)
                print(f"  *** NEW BEST: {best_bound:.6f} (n=6, m={m}) ***")

    # =====================================================================
    # Summary
    # =====================================================================
    total_elapsed = time.time() - start
    print("\n\n" + "=" * 70)
    print("  RESULTS SUMMARY (GPU)")
    print(f"  Device: {device}")
    print("=" * 70)
    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"\n  {'n':>3} {'m':>6} {'d':>3} {'configs':>12} {'corr':>10} "
          f"{'bound':>10} {'time':>8} {'rate':>12}")
    print(f"  {'-'*3} {'-'*6} {'-'*3} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")
    for r in results:
        marker = " <-- BEST" if (r['n_half'], r['m']) == best_config else ""
        print(f"  {r['n_half']:>3} {r['m']:>6} {r['d']:>3} "
              f"{fmt_count(r['n_configs']):>12} {r['correction']:>10.6f} "
              f"{r['bound']:>10.6f} {r['elapsed']:>7.1f}s "
              f"{fmt_count(int(r['rate'])):>11}/s{marker}")

    print(f"\n  *** BEST PROVEN BOUND: C_{{1a}} >= {best_bound:.6f} ***")
    print(f"  *** Achieved at n_half={best_config[0]}, m={best_config[1]} ***")

    # Compare with CPU baseline if available
    cpu_best = 1.1252277777777775  # from run_1hour_20260215_092522.json
    if best_bound > cpu_best:
        print(f"\n  GPU improvement over CPU baseline: {best_bound:.6f} vs {cpu_best:.6f} "
              f"(+{best_bound - cpu_best:.6f})")
    else:
        print(f"\n  CPU baseline: {cpu_best:.6f}")

    # Save results
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'data/run_1hour_gpu_{timestamp}.json'
    out = {
        'timestamp': timestamp,
        'device': device,
        'backend': 'gpu',
        'total_elapsed': total_elapsed,
        'best_bound': best_bound,
        'best_config': {'n_half': best_config[0], 'm': best_config[1]},
        'cpu_baseline_bound': cpu_best,
        'runs': results,
    }
    # Convert large ints to strings for JSON
    for r in out['runs']:
        r['n_configs'] = str(r['n_configs'])
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
