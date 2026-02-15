"""GPU solver API matching the CPU interface.

Provides gpu_find_best_bound_direct() and gpu_run_single_level() with
the same signatures as the CPU versions in solvers.py.
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pruning import correction, count_compositions, asymmetry_threshold
from test_values import compute_test_value_single
from gpu import wrapper


def gpu_find_best_bound_direct(n_half, m, verbose=True):
    """GPU version: compute best provable lower bound in a single pass.

    Returns
    -------
    float : best provable lower bound on C_{1a}.
    """
    d = 2 * n_half
    S = 4 * n_half * m
    n_total = count_compositions(d, S)
    corr = correction(m)

    # Seed running min from uniform composition
    a_uniform = np.full(d, float(4 * n_half) / d)
    init_tv = compute_test_value_single(a_uniform, n_half)
    min_eff = init_tv - corr

    if verbose:
        dev = wrapper.get_device_name()
        print(f"GPU Single-pass: n={n_half}, m={m}, d={d}, S={S}")
        print(f"  Device: {dev}")
        print(f"  Grid points: {n_total:,}")
        print(f"  Correction: {corr:.6f}")
        print(f"  Initial bound (uniform): {min_eff:.6f}")

    t0 = time.time()

    gpu_min_eff, gpu_min_config = wrapper.find_best_bound_direct(
        d, S, n_half, m, min_eff)

    if gpu_min_eff < min_eff:
        min_eff = gpu_min_eff
        min_config = gpu_min_config
    else:
        min_config = None

    elapsed = time.time() - t0

    if verbose:
        print(f"  Completed in {elapsed:.3f}s")
        print(f"  >>> PROVEN: C_{{1a}} >= {min_eff:.6f} <<<")
        if min_config is not None:
            a_cfg = min_config.astype(np.float64) / m
            print(f"  Minimizer (a-coords): {a_cfg}")

    return min_eff


def gpu_run_single_level(n_half, m, c_target, verbose=True):
    """GPU version: run the branch-and-prune at a single discretization level.

    Returns
    -------
    dict with keys: proven, c_proven, n_survivors, min_test_val,
                    min_test_config, stats
    """
    d = 2 * n_half
    S = 4 * n_half * m
    n_total = count_compositions(d, S)
    corr = correction(m)
    prune_target = c_target + corr
    asym_thresh = asymmetry_threshold(c_target)

    if verbose:
        dev = wrapper.get_device_name()
        print(f"GPU Level n={n_half}, m={m}: d={d} bins, S={S}")
        print(f"  Device: {dev}")
        print(f"  Grid points: {n_total:,}")
        print(f"  Correction: {corr:.6f}")
        print(f"  Target: C_{{1a}} >= {c_target:.4f}")
        print(f"  Prune threshold: test_val > {prune_target:.6f}")
        print(f"  Asymmetry threshold: left_frac >= {asym_thresh:.4f}")

    t0 = time.time()

    result = wrapper.run_single_level(d, S, n_half, m, c_target)

    elapsed = time.time() - t0

    n_pruned_asym = result['n_pruned_asym']
    n_pruned_test = result['n_pruned_test']
    n_survived = result['n_survivors']
    n_processed = n_pruned_asym + n_pruned_test + n_survived

    if n_survived > 0:
        min_test_val = result['min_test_val']
        min_test_config = result['min_test_config']
    else:
        min_test_val = float('inf')
        min_test_config = None

    proven = n_survived == 0
    c_proven = c_target if proven else None

    if verbose:
        print(f"\n  Completed in {elapsed:.3f}s "
              f"({n_processed:,} phase2 survivors)")
        print(f"  Asymmetry pruned (phase2): {n_pruned_asym:,}")
        print(f"  Test pruned (phase2): {n_pruned_test:,}")
        print(f"  Survivors: {n_survived:,}")
        if proven:
            print(f"  >>> PROVEN: C_{{1a}} >= {c_target:.6f} <<<")
        else:
            print(f"  NOT proven at target {c_target:.4f}")
            if min_test_config is not None:
                a_cfg = min_test_config.astype(np.float64) / m
                print(f"  Min test value: {min_test_val:.6f}")
                print(f"  Min config (a-coords): {a_cfg}")

    return {
        'proven': proven,
        'c_proven': c_proven,
        'n_survivors': n_survived,
        'survivors': np.empty((0, d), dtype=np.int32),
        'min_test_val': min_test_val,
        'min_test_config': min_test_config,
        'stats': {
            'n_half': n_half, 'm': m, 'd': d, 'S': S,
            'n_total': n_total, 'n_processed': n_processed,
            'n_pruned_asym': n_pruned_asym,
            'n_pruned_test': n_pruned_test,
            'n_survived': n_survived,
            'elapsed': elapsed,
            'prune_target': prune_target,
            'correction': corr,
            'asym_threshold': asym_thresh,
            'backend': 'gpu',
        },
    }
