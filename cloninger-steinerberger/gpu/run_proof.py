"""Multi-level GPU proof runner for the autoconvolution constant.

Usage:
    python -m cloninger-steinerberger.gpu.run_proof [--c_target C] [--m M] [--n_half N]
                                                     [--max_levels L] [--time_budget T]

Default: n_half=2, m=200, c_target=1.30 (d=4 at Level 0, tighter correction at 0.010).
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'archive', 'cpu'))
from pruning import correction, count_compositions, asymmetry_threshold
from gpu import wrapper
from gpu.solvers import gpu_find_best_bound_direct, gpu_run_single_level


def run_multilevel_proof(n_half, m, c_target, max_levels=5, time_budget_sec=0,
                         max_survivors_per_level=200_000_000, verbose=True):
    """Run a multi-level branch-and-prune proof.

    Parameters
    ----------
    n_half : int
        Initial half-dimension (d = 2*n_half at Level 0).
    m : int
        Grid resolution (S = m).
    c_target : float
        Target lower bound to prove.
    max_levels : int
        Maximum number of refinement levels.
    time_budget_sec : float
        Time budget per level in seconds (0 = no limit).
    max_survivors_per_level : int
        Max survivors to extract per level.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with proof result.
    """
    d0 = 2 * n_half
    corr = correction(m)
    prune_target = c_target + corr

    print("=" * 70)
    print(f"MULTI-LEVEL PROOF: C_{{1a}} >= {c_target}")
    print("=" * 70)
    print(f"  n_half={n_half}, m={m}, d0={d0}")
    print(f"  Correction: {corr:.6f}")
    print(f"  Effective threshold: {prune_target:.6f}")
    print(f"  Max levels: {max_levels}")
    if time_budget_sec > 0:
        print(f"  Time budget per level: {time_budget_sec:.0f}s")
    print()

    proof_start = time.time()
    level_results = []

    # ---- Level 0: enumerate all d0-bin compositions ----
    print(f"{'='*70}")
    print(f"LEVEL 0: d={d0}, n_half={n_half}")
    print(f"{'='*70}")

    result = gpu_run_single_level(
        n_half, m, c_target,
        verbose=True,
        extract_survivors=True,
        stream_to_disk=True)

    level_results.append({
        'level': 0,
        'd': d0,
        'n_half': n_half,
        **result['stats'],
    })

    if result['proven']:
        total_time = time.time() - proof_start
        print(f"\n{'='*70}")
        print(f"PROVEN at Level 0: C_{{1a}} >= {c_target} in {total_time:.1f}s")
        print(f"{'='*70}")
        return _build_result(True, c_target, level_results, total_time)

    # Load survivors for refinement
    survivors = _load_survivors(result, d0)
    print(f"\nLevel 0 survivors: {len(survivors):,}")

    # ---- Refinement levels ----
    for level in range(1, max_levels + 1):
        d_parent = d0 * (2 ** (level - 1))
        d_child = 2 * d_parent
        n_half_child = n_half * (2 ** level)

        print(f"\n{'='*70}")
        print(f"LEVEL {level}: d_parent={d_parent} -> d_child={d_child}, "
              f"n_half={n_half_child}")
        print(f"  Parents: {len(survivors):,}")
        print(f"{'='*70}")

        if len(survivors) == 0:
            total_time = time.time() - proof_start
            print(f"\nPROVEN at Level {level-1}: no survivors to refine.")
            return _build_result(True, c_target, level_results, total_time)

        t0 = time.time()

        # Compute max survivors from GPU memory
        max_surv = wrapper.max_survivors_for_dim(d_child)
        max_surv = min(max_surv, max_survivors_per_level)

        refine_result = wrapper.refine_parents(
            d_parent=d_parent,
            parent_configs_array=survivors,
            m=m,
            c_target=c_target,
            max_survivors=max_surv,
            time_budget_sec=time_budget_sec)

        elapsed = time.time() - t0

        total_refs = (refine_result['total_asym'] +
                      refine_result['total_test'] +
                      refine_result['total_survivors'])
        rate = total_refs / elapsed if elapsed > 0 else 0

        print(f"\n  Level {level} completed in {elapsed:.1f}s")
        print(f"  Total refinements: {total_refs:,.0f}")
        print(f"  Throughput: {rate:,.0f} refs/sec")
        print(f"  Asymmetry pruned: {refine_result['total_asym']:,}")
        print(f"  Test pruned: {refine_result['total_test']:,}")
        print(f"  Survivors: {refine_result['total_survivors']:,}")
        print(f"  Extracted: {refine_result['n_extracted']:,}")
        if refine_result['timed_out']:
            print(f"  WARNING: timed out")

        level_results.append({
            'level': level,
            'd_parent': d_parent,
            'd_child': d_child,
            'n_half_child': n_half_child,
            'n_parents': len(survivors),
            'total_refs': total_refs,
            'total_asym': refine_result['total_asym'],
            'total_test': refine_result['total_test'],
            'total_survivors': refine_result['total_survivors'],
            'n_extracted': refine_result['n_extracted'],
            'elapsed': elapsed,
            'rate': rate,
            'timed_out': refine_result['timed_out'],
        })

        if refine_result['total_survivors'] == 0:
            total_time = time.time() - proof_start
            print(f"\n{'='*70}")
            print(f"PROVEN at Level {level}: C_{{1a}} >= {c_target} in {total_time:.1f}s")
            print(f"{'='*70}")
            return _build_result(True, c_target, level_results, total_time)

        # Prepare survivors for next level
        survivors = refine_result['survivor_configs']
        if len(survivors) == 0 and refine_result['total_survivors'] > 0:
            print(f"\n  WARNING: survivors exist but none extracted "
                  f"(buffer overflow or timeout). Cannot continue.")
            break

        print(f"\n  Carrying {len(survivors):,} survivors to Level {level+1}")

    # Exhausted all levels without proving
    total_time = time.time() - proof_start
    final_survs = refine_result['total_survivors'] if 'refine_result' in dir() else result['n_survivors']
    print(f"\n{'='*70}")
    print(f"NOT PROVEN after {max_levels} levels ({total_time:.1f}s)")
    print(f"Remaining survivors: {final_survs:,}")
    print(f"{'='*70}")
    return _build_result(False, c_target, level_results, total_time)


def _load_survivors(result, d):
    """Load survivors from Level 0 result (in-memory or disk)."""
    if result.get('survivor_file') and os.path.exists(result['survivor_file']):
        return wrapper.load_survivors_chunk(result['survivor_file'], d)
    else:
        return result['survivors']


def _build_result(proven, c_target, level_results, total_time):
    """Build the final result dict."""
    return {
        'proven': proven,
        'c_target': c_target,
        'levels': level_results,
        'total_time': total_time,
    }


def save_proof_result(result, n_half, m):
    """Save proof result to a JSON file in data/."""
    os.makedirs('data', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    filename = os.path.join('data', f'gpu_proof_{ts}.json')

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        'timestamp': ts,
        'n_half': n_half,
        'm': m,
        **{k: convert(v) for k, v in result.items()},
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\nResult saved to {filename}")
    return filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-level GPU proof for autoconvolution constant')
    parser.add_argument('--c_target', type=float, default=1.30,
                        help='Target lower bound (default: 1.30)')
    parser.add_argument('--m', type=int, default=200,
                        help='Grid resolution (default: 200)')
    parser.add_argument('--n_half', type=int, default=2,
                        help='Initial half-dimension (default: 2, d=4)')
    parser.add_argument('--max_levels', type=int, default=5,
                        help='Maximum refinement levels (default: 5)')
    parser.add_argument('--time_budget', type=float, default=0,
                        help='Time budget per level in seconds (0=unlimited)')
    parser.add_argument('--find_min', action='store_true',
                        help='Run find_best_bound_direct instead of proof')
    args = parser.parse_args()

    # Check GPU
    if not wrapper.is_available():
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    dev = wrapper.get_device_name()
    free_mb = wrapper.get_free_memory() / (1024 * 1024)
    print(f"GPU: {dev}")
    print(f"Free memory: {free_mb:.0f} MB")
    print()

    if args.find_min:
        # Single-pass find minimum effective value
        result = gpu_find_best_bound_direct(args.n_half, args.m, verbose=True)
        print(f"\nBest provable bound: {result:.6f}")
    else:
        # Multi-level proof
        result = run_multilevel_proof(
            n_half=args.n_half,
            m=args.m,
            c_target=args.c_target,
            max_levels=args.max_levels,
            time_budget_sec=args.time_budget)

        save_proof_result(result, args.n_half, args.m)
