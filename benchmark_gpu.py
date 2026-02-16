"""Comprehensive GPU benchmark for Sidon autocorrelation branch-and-prune.

Tests all algorithm steps, measures throughput, pruning effectiveness,
and survivor counts at various target thresholds.

Usage (on RunPod A100):
    python benchmark_gpu.py                # full benchmark
    python benchmark_gpu.py --quick        # quick smoke test (~30s)

Results saved to data/benchmark_{timestamp}.json
"""
import sys
import os
import time
import json
import argparse
import platform
from datetime import datetime

# Import setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger', 'cpu'))

from gpu.wrapper import (
    is_available, get_device_name, get_free_memory,
    find_best_bound_direct, run_single_level, run_single_level_extract,
    refine_parents, max_survivors_for_dim,
)
from pruning import correction, count_compositions, asymmetry_threshold
from test_values import compute_test_value_single
import numpy as np


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt(n):
    if isinstance(n, float):
        if n >= 1e12:
            return f"{n:.2e}"
        elif n >= 1e6:
            return f"{n:,.0f}"
        return f"{n:,.2f}"
    return f"{n:,}"


def bench_find_min(n_half, m, label=""):
    """Benchmark gpu_find_best_bound_direct."""
    d = 2 * n_half
    S = m  # S=m convention
    n_total = count_compositions(d, S)
    corr = correction(m)

    # Seed from uniform
    a_uniform = np.full(d, float(4 * n_half) / d)
    init_tv = compute_test_value_single(a_uniform, n_half)
    init_min_eff = init_tv - corr

    log(f"  find_min D={d} m={m}: {fmt(n_total)} points")
    t0 = time.time()
    min_eff, min_config = find_best_bound_direct(d, S, n_half, m, init_min_eff)
    elapsed = time.time() - t0
    rate = n_total / elapsed if elapsed > 0 else 0

    result = {
        "test": f"find_min_D{d}_{label}" if label else f"find_min_D{d}_m{m}",
        "n_half": n_half, "m": m, "d": d, "S": S,
        "n_total": n_total,
        "elapsed_s": round(elapsed, 4),
        "throughput": round(rate, 0),
        "min_eff": round(min_eff, 8),
        "min_config": min_config.tolist(),
    }
    log(f"    -> {elapsed:.3f}s, {rate:.2e} configs/s, bound={min_eff:.6f}")
    return result


def bench_prove_count(n_half, m, c_target, label=""):
    """Benchmark gpu_run_single_level (count only, no extraction)."""
    d = 2 * n_half
    S = m  # S=m convention
    n_total = count_compositions(d, S)
    corr = correction(m)

    log(f"  prove D={d} m={m} target={c_target}: {fmt(n_total)} points")
    t0 = time.time()
    res = run_single_level(d, S, n_half, m, c_target)
    elapsed = time.time() - t0

    n_processed = res['n_pruned_asym'] + res['n_pruned_test'] + res['n_survivors']
    rate = n_total / elapsed if elapsed > 0 else 0

    result = {
        "test": f"prove_D{d}_{label}" if label else f"prove_D{d}_m{m}_t{c_target}",
        "n_half": n_half, "m": m, "d": d, "S": S, "c_target": c_target,
        "n_total": n_total,
        "elapsed_s": round(elapsed, 4),
        "throughput": round(rate, 0),
        "n_pruned_asym": res['n_pruned_asym'],
        "n_pruned_test": res['n_pruned_test'],
        "n_survivors": res['n_survivors'],
        "pct_asym": round(100 * res['n_pruned_asym'] / max(n_processed, 1), 2),
        "pct_test": round(100 * res['n_pruned_test'] / max(n_processed, 1), 2),
        "pct_surv": round(100 * res['n_survivors'] / max(n_processed, 1), 4),
        "proven": res['n_survivors'] == 0,
    }
    if res['n_survivors'] > 0:
        result["min_test_val"] = round(res['min_test_val'], 8)
    log(f"    -> {elapsed:.3f}s, {rate:.2e}/s, surv={fmt(res['n_survivors'])} "
        f"({result['pct_surv']:.4f}%)")
    return result


def bench_prove_extract(n_half, m, c_target, label=""):
    """Benchmark gpu_run_single_level with survivor extraction."""
    d = 2 * n_half
    S = m  # S=m convention
    n_total = count_compositions(d, S)
    max_surv = max_survivors_for_dim(d)

    log(f"  extract D={d} m={m} target={c_target}: buf={fmt(max_surv)}")
    t0 = time.time()
    res = run_single_level_extract(d, S, n_half, m, c_target, max_survivors=max_surv)
    elapsed = time.time() - t0

    n_processed = res['n_pruned_asym'] + res['n_pruned_test'] + res['n_survivors']
    rate = n_total / elapsed if elapsed > 0 else 0

    result = {
        "test": f"extract_D{d}_{label}" if label else f"extract_D{d}_m{m}_t{c_target}",
        "n_half": n_half, "m": m, "d": d, "S": S, "c_target": c_target,
        "n_total": n_total,
        "elapsed_s": round(elapsed, 4),
        "throughput": round(rate, 0),
        "n_survivors": res['n_survivors'],
        "n_extracted": res['n_extracted'],
        "buffer_overflow": res['n_survivors'] > res['n_extracted'],
        "pct_surv": round(100 * res['n_survivors'] / max(n_processed, 1), 4),
    }
    log(f"    -> {elapsed:.3f}s, surv={fmt(res['n_survivors'])}, "
        f"extracted={fmt(res['n_extracted'])}")
    return result, res.get('survivor_configs', None)


def bench_refine(parents, d_parent, m, c_target, max_surv=1000000, time_budget=60, label=""):
    """Benchmark gpu_refine_parents."""
    num_parents = parents.shape[0]
    d_child = 2 * d_parent

    # Estimate total refinements
    refs_per = np.prod(2.0 * parents.astype(np.float64) + 1.0, axis=1)
    total_refs = float(refs_per.sum())

    log(f"  refine d_parent={d_parent}->d_child={d_child}: "
        f"{fmt(num_parents)} parents, ~{total_refs:.2e} refinements")

    t0 = time.time()
    res = refine_parents(
        d_parent=d_parent,
        parent_configs_array=parents,
        m=m,
        c_target=c_target,
        max_survivors=max_surv,
        time_budget_sec=time_budget,
    )
    elapsed = time.time() - t0

    ref_rate = total_refs / elapsed if elapsed > 0 else 0

    result = {
        "test": f"refine_{label}" if label else f"refine_d{d_parent}_to_d{d_child}",
        "d_parent": d_parent, "d_child": d_child, "m": m, "c_target": c_target,
        "num_parents": num_parents,
        "total_refinements": total_refs,
        "elapsed_s": round(elapsed, 4),
        "throughput": round(ref_rate, 0),
        "total_asym": res['total_asym'],
        "total_test": res['total_test'],
        "total_survivors": res['total_survivors'],
        "n_extracted": res['n_extracted'],
        "timed_out": res['timed_out'],
    }
    log(f"    -> {elapsed:.3f}s, {ref_rate:.2e} refs/s, "
        f"surv={fmt(res['total_survivors'])}, timed_out={res['timed_out']}")
    return result, res.get('survivor_configs', None)


def make_synthetic_parents(d_parent, m, num_parents, S_parent):
    """Generate synthetic parent configs for refinement benchmarks."""
    rng = np.random.RandomState(42)
    parents = np.zeros((num_parents, d_parent), dtype=np.int32)
    for i in range(num_parents):
        # Generate random composition summing to S_parent
        cuts = sorted(rng.choice(S_parent + d_parent - 1, d_parent - 1, replace=False))
        prev = 0
        for j, c in enumerate(cuts):
            parents[i, j] = c - prev - j
            prev = c - j
        parents[i, d_parent - 1] = S_parent - parents[i, :d_parent - 1].sum()
    return parents


def main():
    parser = argparse.ArgumentParser(description='GPU benchmark suite')
    parser.add_argument('--quick', action='store_true', help='Quick smoke test (~30s)')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"benchmark_{timestamp}.json")

    results = {
        "timestamp": timestamp,
        "platform": platform.system(),
        "quick_mode": args.quick,
        "benchmarks": [],
    }

    log("=" * 64)
    log("GPU BENCHMARK SUITE — Sidon Autocorrelation")
    log("=" * 64)

    if not is_available():
        log("ERROR: No CUDA GPU available")
        sys.exit(1)

    dev = get_device_name()
    free_mem = get_free_memory()
    log(f"Device: {dev}")
    log(f"Free memory: {free_mem / (1024**3):.1f} GB")
    results["device"] = dev
    results["free_memory_gb"] = round(free_mem / (1024**3), 2)

    # ================================================================
    #  1. find_best_bound_direct benchmarks
    # ================================================================
    log("")
    log("=" * 64)
    log("SECTION 1: find_best_bound_direct")
    log("=" * 64)

    # D=4 benchmarks
    for m_val in ([10, 30] if args.quick else [10, 30, 50, 80]):
        r = bench_find_min(n_half=2, m=m_val, label=f"m{m_val}")
        results["benchmarks"].append(r)

    # D=6 benchmarks
    for m_val in ([10, 20] if args.quick else [10, 20, 30, 40]):
        r = bench_find_min(n_half=3, m=m_val, label=f"m{m_val}")
        results["benchmarks"].append(r)

    # ================================================================
    #  2. run_single_level (count only) at various targets
    # ================================================================
    log("")
    log("=" * 64)
    log("SECTION 2: run_single_level (count mode)")
    log("=" * 64)

    # D=6, m=50: the production configuration, various targets
    targets = [1.28, 1.25, 1.22, 1.20, 1.15, 1.10]
    if args.quick:
        targets = [1.28, 1.20, 1.10]

    log(f"  D=6, m=50 survivor counts at various c_target:")
    for tgt in targets:
        r = bench_prove_count(n_half=3, m=50, c_target=tgt, label=f"t{tgt}")
        results["benchmarks"].append(r)

    # D=4, m=100 for comparison
    if not args.quick:
        log(f"\n  D=4, m=100:")
        r = bench_prove_count(n_half=2, m=100, c_target=1.20, label="m100_t1.20")
        results["benchmarks"].append(r)

    # ================================================================
    #  3. run_single_level with extraction
    # ================================================================
    log("")
    log("=" * 64)
    log("SECTION 3: Survivor extraction")
    log("=" * 64)

    # Extract at m=50, target=1.20 (the production case)
    # Use smaller m in quick mode
    if args.quick:
        ext_m = 30
        ext_tgt = 1.20
    else:
        ext_m = 50
        ext_tgt = 1.20

    r_ext, survivors_6 = bench_prove_extract(
        n_half=3, m=ext_m, c_target=ext_tgt, label=f"m{ext_m}_t{ext_tgt}")
    results["benchmarks"].append(r_ext)

    # Also extract at an easier target for refinement testing
    if not args.quick:
        r_ext2, survivors_easy = bench_prove_extract(
            n_half=3, m=50, c_target=1.28, label="m50_t1.28")
        results["benchmarks"].append(r_ext2)
    else:
        r_ext2, survivors_easy = bench_prove_extract(
            n_half=3, m=30, c_target=1.28, label="m30_t1.28")
        results["benchmarks"].append(r_ext2)

    # ================================================================
    #  4. Refinement benchmarks
    # ================================================================
    log("")
    log("=" * 64)
    log("SECTION 4: Refinement (parent -> child)")
    log("=" * 64)

    # Use real survivors if we have them, otherwise synthetic
    if survivors_easy is not None and len(survivors_easy) > 0:
        # Use a subset for manageable runtime
        n_use = min(len(survivors_easy), 1000 if args.quick else 10000)
        test_parents = survivors_easy[:n_use].copy()
        log(f"  Using {n_use} real D=6 survivors for refinement test")
    else:
        # Synthetic parents
        n_use = 100 if args.quick else 1000
        S_parent = 4 * 3 * 50  # n=3, m=50
        test_parents = make_synthetic_parents(6, 50, n_use, S_parent)
        log(f"  Using {n_use} synthetic D=6 parents")

    m_refine = ext_m  # must match the m used for extraction
    tgt_refine = 1.28
    r_ref, _ = bench_refine(
        test_parents, d_parent=6, m=m_refine, c_target=tgt_refine,
        max_surv=1000000, time_budget=30 if args.quick else 60,
        label=f"d6_to_d12_t{tgt_refine}")
    results["benchmarks"].append(r_ref)

    # ================================================================
    #  5. Memory usage summary
    # ================================================================
    log("")
    log("=" * 64)
    log("SECTION 5: Memory usage")
    log("=" * 64)

    free_after = get_free_memory()
    log(f"  Free memory after all tests: {free_after / (1024**3):.1f} GB")
    results["free_memory_after_gb"] = round(free_after / (1024**3), 2)

    # Compute memory requirements for production runs
    mem_info = []
    for d_val, label in [(6, "Level0_D6"), (12, "Level1_D12"), (24, "Level2_D24")]:
        for surv_count in [1_000_000, 100_000_000, 1_000_000_000]:
            bytes_needed = surv_count * d_val * 4
            mem_info.append({
                "label": f"{label}_{surv_count//1_000_000}M_survivors",
                "d": d_val,
                "n_survivors": surv_count,
                "memory_gb": round(bytes_needed / (1024**3), 2),
            })
    results["memory_estimates"] = mem_info

    for m in mem_info:
        log(f"  {m['label']}: {m['memory_gb']:.2f} GB")

    # ================================================================
    #  Summary
    # ================================================================
    log("")
    log("=" * 64)
    log("SUMMARY")
    log("=" * 64)

    # Print key results table
    log(f"{'Test':<40} {'Time':>8} {'Throughput':>14} {'Survivors':>14}")
    log("-" * 80)
    for b in results["benchmarks"]:
        test_name = b["test"][:40]
        elapsed = f"{b['elapsed_s']:.3f}s"
        tp = f"{b.get('throughput', 0):.2e}"
        surv = fmt(b.get('n_survivors', b.get('total_survivors', '-')))
        log(f"{test_name:<40} {elapsed:>8} {tp:>14} {surv:>14}")

    # Save results
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
