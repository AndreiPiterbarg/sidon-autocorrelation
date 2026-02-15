"""
curriculum_core.py — Curriculum learning helpers for Sidon autocorrelation.

Imports core math from sidon_core. No Modal dependency.

Adds:
  - keep_diverse: diversity-filtered solution selection
  - explore_single_P: massive exploration at one P value
  - cascade_candidate: polish one upsampled candidate at a cascade level
"""

import time
import numpy as np
from joblib import Parallel, delayed
from sidon_core import (
    _hybrid_single_restart,
    hybrid_single_restart_dispatch,
    make_inits,
    autoconv_coeffs,
    warmup as _core_warmup,
)

CURRICULUM_STRATEGIES = [
    'dirichlet_sparse', 'gaussian_peak', 'bimodal',
    'boundary_heavy', 'symmetric_dirichlet', 'warm_perturb',
]


def keep_diverse(solutions, threshold=0.1, max_keep=10):
    """Keep up to max_keep diverse solutions from a sorted (best-first) list.

    Args:
        solutions: list of (val, x_array) sorted by val ascending (best first)
        threshold: minimum L2 distance between kept solutions
        max_keep: max number to keep

    Returns:
        list of (val, x_array) that are mutually diverse
    """
    if not solutions:
        return []
    kept = [solutions[0]]
    for val, x in solutions[1:]:
        if len(kept) >= max_keep:
            break
        if all(np.linalg.norm(x - xk) >= threshold for _, xk in kept):
            kept.append((val, x))
    return kept


def explore_single_P(P, strategies, n_restarts_per_strategy,
                     beta_schedule, n_iters_lse, n_iters_polyak,
                     diversity_threshold, top_k, n_jobs=-1, seed=None):
    """
    Massive exploration at one P value: all strategies x n_restarts.

    Returns dict with:
        P, best_val, best_x (list), diverse [(val, x_list), ...], n_total
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(beta_schedule, dtype=np.float64)

    all_solutions = []  # list of (val, x_array)

    for strat_idx, strategy in enumerate(strategies):
        # For warm_perturb, use the best solution found so far at this P
        warm_x = None
        if strategy == 'warm_perturb' and all_solutions:
            warm_x = min(all_solutions, key=lambda s: s[0])[1]

        inits = make_inits(strategy, P, n_restarts_per_strategy, rng, warm_x=warm_x)

        t0 = time.time()
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(hybrid_single_restart_dispatch)(
                inits[i], P, beta_arr, n_iters_lse, n_iters_polyak
            )
            for i in range(n_restarts_per_strategy)
        )
        dt = time.time() - t0

        vals = []
        for lse_v, pol_v, pol_x in results:
            all_solutions.append((float(pol_v), pol_x))
            vals.append(pol_v)

        best_idx = int(np.argmin(vals))
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] [{strat_idx+1}/{len(strategies)}] {strategy:25s}: "
              f"best={vals[best_idx]:.6f}  median={np.median(vals):.6f}  "
              f"std={np.std(vals):.4f}  time={dt:.1f}s")

    # Sort and diversity-filter
    all_solutions.sort(key=lambda s: s[0])
    scaled_threshold = diversity_threshold * np.sqrt(50.0 / P)
    diverse = keep_diverse(all_solutions, threshold=scaled_threshold, max_keep=top_k)

    print(f"  Overall best: {all_solutions[0][0]:.6f}, "
          f"diverse kept: {len(diverse)}/{len(all_solutions)}")

    return {
        'P': P,
        'best_val': float(all_solutions[0][0]),
        'best_x': all_solutions[0][1].tolist(),
        'diverse': [(float(v), x.tolist()) for v, x in diverse],
        'n_total': len(all_solutions),
    }


def cascade_candidate(x_up, P_target, n_warm_restarts,
                      beta_schedule_direct, beta_schedule_warm,
                      n_iters_lse, n_iters_polyak,
                      n_jobs=-1, seed=None):
    """
    Optimize one upsampled candidate at a cascade level.

    Runs:
      1. Direct polish of x_up (BETA_ULTRA schedule)
      2. n_warm_restarts warm perturbations around x_up (BETA_HEAVY schedule)

    Returns list of (val, x_list) for all restarts.
    """
    rng = np.random.default_rng(seed)
    beta_direct = np.array(beta_schedule_direct, dtype=np.float64)
    beta_warm = np.array(beta_schedule_warm, dtype=np.float64)

    all_results = []

    # 1. Direct polish of the upsampled solution
    _, pol_v, pol_x = hybrid_single_restart_dispatch(
        x_up, P_target, beta_direct, n_iters_lse, n_iters_polyak
    )
    all_results.append((float(pol_v), pol_x.tolist()))

    # 2. Warm perturbations around this candidate
    if n_warm_restarts > 0:
        perturb_inits = make_inits(
            'warm_perturb', P_target, n_warm_restarts, rng, warm_x=x_up
        )
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(hybrid_single_restart_dispatch)(
                perturb_inits[i], P_target, beta_warm, n_iters_lse, n_iters_polyak
            )
            for i in range(n_warm_restarts)
        )
        for lse_v, pol_v, pol_x in results:
            all_results.append((float(pol_v), pol_x.tolist()))

    return all_results


def warmup():
    """Trigger Numba compilation for all JIT functions."""
    _core_warmup()
