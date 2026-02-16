"""GPU solver API matching the CPU interface.

Provides gpu_find_best_bound_direct() and gpu_run_single_level() with
the same signatures as the CPU versions in solvers.py.
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpu'))
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
    S = m  # S=m convention: integer coords sum to m (not 4nm)
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
            # With S=m: a_j = (4n/m) * c_j
            a_cfg = min_config.astype(np.float64) * (4 * n_half) / m
            print(f"  Minimizer (a-coords): {a_cfg}")

    return min_eff


def gpu_run_single_level(n_half, m, c_target, verbose=True, extract_survivors=False,
                         max_survivors=2000000000, stream_to_disk=None,
                         survivor_file=None):
    """GPU version: run the branch-and-prune at a single discretization level.

    Parameters
    ----------
    extract_survivors : bool
        If True, extract actual survivor configurations (for multi-level refinement).
    max_survivors : int
        Maximum number of survivor configs to extract (default 2B, sized for A100).
    stream_to_disk : bool or None
        If True, force streamed extraction to disk. If None (default), auto-select:
        use streamed mode for D>=6 and m>=20 when extracting survivors.
    survivor_file : str or None
        Path for the binary survivor file when streaming. If None, auto-generated
        in data/ directory.

    Returns
    -------
    dict with keys: proven, c_proven, n_survivors, min_test_val,
                    min_test_config, survivors, stats
    When streamed mode is used, 'survivors' is np.empty and 'survivor_file'
    + 'n_extracted' are provided instead.
    """
    d = 2 * n_half
    S = m  # S=m convention: integer coords sum to m (not 4nm)
    n_total = count_compositions(d, S)
    corr = correction(m)
    prune_target = c_target + corr
    asym_thresh = asymmetry_threshold(c_target)

    # Decide streaming mode
    use_streamed = False
    if extract_survivors:
        if stream_to_disk is True:
            use_streamed = True
        elif stream_to_disk is None:
            # Auto: stream for D>=6, m>=20 (likely >100M survivors)
            use_streamed = (d >= 6 and m >= 20)

    if verbose:
        dev = wrapper.get_device_name()
        print(f"GPU Level n={n_half}, m={m}: d={d} bins, S={S}")
        print(f"  Device: {dev}")
        print(f"  Grid points: {n_total:,}")
        print(f"  Correction: {corr:.6f}")
        print(f"  Target: C_{{1a}} >= {c_target:.4f}")
        print(f"  Prune threshold: test_val > {prune_target:.6f}")
        print(f"  Asymmetry threshold: left_frac >= {asym_thresh:.4f}")
        if extract_survivors:
            if use_streamed:
                print(f"  Extraction mode: STREAMED TO DISK (d={d}, m={m})")
            else:
                print(f"  Extraction mode: IN-MEMORY (max {max_survivors:,})")

    t0 = time.time()

    if extract_survivors and use_streamed:
        # Streamed extraction to disk
        if survivor_file is None:
            import time as _time
            ts = _time.strftime('%Y%m%d_%H%M%S')
            os.makedirs('data', exist_ok=True)
            survivor_file = os.path.join('data', f'survivors_d{d}_m{m}_{ts}.bin')

        if verbose:
            print(f"  Survivor file: {survivor_file}")

        result = wrapper.run_single_level_extract_streamed(
            d, S, n_half, m, c_target, survivor_file)

    elif extract_survivors:
        result = wrapper.run_single_level_extract(d, S, n_half, m, c_target,
                                                   max_survivors=max_survivors)
    else:
        result = wrapper.run_single_level(d, S, n_half, m, c_target)

    elapsed = time.time() - t0

    n_fp32_skipped = result.get('n_fp32_skipped', 0)
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

    if extract_survivors and use_streamed and not proven:
        survivors = np.empty((0, d), dtype=np.int32)  # on disk, not in memory
        n_extracted = result['n_extracted']
        out_survivor_file = result['survivor_file']
    elif extract_survivors and not proven:
        survivors = result['survivor_configs']
        n_extracted = result['n_extracted']
        out_survivor_file = None
    else:
        survivors = np.empty((0, d), dtype=np.int32)
        n_extracted = 0
        out_survivor_file = None

    if verbose:
        print(f"\n  Completed in {elapsed:.3f}s "
              f"({n_processed:,} phase2 survivors)")
        print(f"  FP32 pre-checks skipped: {n_fp32_skipped:,}")
        print(f"  Asymmetry pruned (FP64 final): {n_pruned_asym:,}")
        print(f"  Test pruned (phase2): {n_pruned_test:,}")
        print(f"  Survivors: {n_survived:,}")
        if extract_survivors and n_extracted > 0:
            print(f"  Extracted: {n_extracted:,} configs")
            if use_streamed:
                print(f"  Streamed to: {out_survivor_file}")
            elif n_extracted < n_survived:
                print(f"  WARNING: buffer overflow, only {n_extracted}/{n_survived} extracted")
        if proven:
            print(f"  >>> PROVEN: C_{{1a}} >= {c_target:.6f} <<<")
        else:
            print(f"  NOT proven at target {c_target:.4f}")
            if min_test_config is not None:
                # With S=m: a_j = (4n/m) * c_j
                a_cfg = min_test_config.astype(np.float64) * (4 * n_half) / m
                print(f"  Min test value: {min_test_val:.6f}")
                print(f"  Min config (a-coords): {a_cfg}")

    return {
        'proven': proven,
        'c_proven': c_proven,
        'n_survivors': n_survived,
        'survivors': survivors,
        'survivor_file': out_survivor_file,
        'min_test_val': min_test_val,
        'min_test_config': min_test_config,
        'stats': {
            'n_half': n_half, 'm': m, 'd': d, 'S': S,
            'n_total': n_total, 'n_processed': n_processed,
            'n_fp32_skipped': n_fp32_skipped,
            'n_pruned_asym': n_pruned_asym,
            'n_pruned_test': n_pruned_test,
            'n_survived': n_survived,
            'n_extracted': n_extracted,
            'elapsed': elapsed,
            'prune_target': prune_target,
            'correction': corr,
            'asym_threshold': asym_thresh,
            'backend': 'gpu',
            'streamed': use_streamed,
        },
    }
