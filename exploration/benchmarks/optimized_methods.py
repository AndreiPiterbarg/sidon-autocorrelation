"""
Speed-optimized versions of the 5 Sidon optimization methods.

Optimizations:
1. Uses _hybrid_single_restart_fast (lazy k_star Polyak, reduced LSE tracking)
2. Warm-start methods use a shorter beta schedule (skip early smooth stages)
3. Warm-start methods use fewer LSE iterations (solution is already close)
"""

import sys, os, time, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")

from sidon_core import autoconv_coeffs, BETA_HEAVY, BETA_ULTRA
from experiments.speed_tests.fast_core import _hybrid_single_restart_fast

P = 200

# Shorter beta schedules for warm-started methods
# Skip early low-beta stages since warm solution is already well-optimized
BETA_WARM = [42, 65, 100, 150, 230, 350, 500, 750, 1000, 1500, 2000, 3000]
BETA_WARM_ULTRA = [42, 65, 100, 150, 230, 350, 500, 750, 1000, 1500, 2000, 3000, 4000]


def verify_solution(x):
    c = autoconv_coeffs(x, P)
    return float(np.max(c))


def load_best_solution():
    fname = os.path.join(EXPERIMENTS_DIR, "solution_final_final_baseline_P200.json")
    if os.path.exists(fname):
        with open(fname) as f:
            data = json.load(f)
        return np.array(data['x'])
    return np.ones(P) / P


# ============================================================================
# Optimized Method 1: Iterated Warm Restart
# Uses WARM beta schedule (fewer stages) + reduced LSE iters
# ============================================================================

def run_iterated_warm_fast(time_budget, seed=42):
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_WARM_ULTRA, dtype=np.float64)
    warm_x = load_best_solution()
    best_val = verify_solution(warm_x)
    best_x = warm_x.copy()
    n = 0
    scales = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]

    t0 = time.time()
    while time.time() - t0 < time_budget:
        scale = scales[n % len(scales)]
        x_init = warm_x + scale * rng.standard_normal(P) * np.mean(warm_x)
        x_init = np.maximum(x_init, 0.0)
        if x_init.sum() < 1e-12:
            x_init = rng.dirichlet(np.ones(P))
        else:
            x_init /= x_init.sum()
        _, pol_v, x = _hybrid_single_restart_fast(
            x_init, P, beta_arr, 10000, 200000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
            warm_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# Optimized Method 2: Mirrored Sampling
# Uses WARM beta schedule + reduced LSE iters
# ============================================================================

def run_mirrored_sampling_fast(time_budget, seed=42):
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_WARM, dtype=np.float64)
    warm_x = load_best_solution()
    best_val = verify_solution(warm_x)
    best_x = warm_x.copy()
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        scale = rng.uniform(0.05, 0.8)
        z = scale * rng.standard_normal(P) * np.mean(warm_x)

        x_plus = warm_x + z
        x_plus = np.maximum(x_plus, 0.0)
        if x_plus.sum() > 1e-12:
            x_plus /= x_plus.sum()
        else:
            x_plus = rng.dirichlet(np.ones(P))

        _, pol_v1, x1 = _hybrid_single_restart_fast(
            x_plus, P, beta_arr, 10000, 200000)
        n += 1
        if pol_v1 < best_val:
            best_val = pol_v1
            best_x = x1.copy()
            warm_x = x1.copy()

        if time.time() - t0 >= time_budget:
            break

        x_minus = warm_x - z
        x_minus = np.maximum(x_minus, 0.0)
        if x_minus.sum() > 1e-12:
            x_minus /= x_minus.sum()
        else:
            x_minus = rng.dirichlet(np.ones(P))

        _, pol_v2, x2 = _hybrid_single_restart_fast(
            x_minus, P, beta_arr, 10000, 200000)
        n += 1
        if pol_v2 < best_val:
            best_val = pol_v2
            best_x = x2.copy()
            warm_x = x2.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# Optimized Method 3: Adaptive Perturbation
# Uses WARM beta schedule + reduced LSE iters
# ============================================================================

def run_adaptive_perturbation_fast(time_budget, seed=42):
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_WARM, dtype=np.float64)
    warm_x = load_best_solution()
    best_val = verify_solution(warm_x)
    best_x = warm_x.copy()
    n = 0

    n_arms = 8
    scales = np.logspace(-2, 0.3, n_arms)
    successes = np.ones(n_arms)
    failures = np.ones(n_arms)

    t0 = time.time()
    while time.time() - t0 < time_budget:
        probs = np.array([rng.beta(s, f) for s, f in zip(successes, failures)])
        arm = np.argmax(probs)
        scale = scales[arm]

        x_init = warm_x + scale * rng.standard_normal(P) * np.mean(warm_x)
        x_init = np.maximum(x_init, 0.0)
        if x_init.sum() < 1e-12:
            x_init = rng.dirichlet(np.ones(P))
        else:
            x_init /= x_init.sum()

        _, pol_v, x = _hybrid_single_restart_fast(
            x_init, P, beta_arr, 10000, 200000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
            warm_x = x.copy()
            successes[arm] += 1
        else:
            failures[arm] += 1

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# Optimized Method 4: Heavy-Tail Elite v2
# Cold start — uses full BETA_ULTRA + fast core
# ============================================================================

def run_heavy_elite_v2_fast(time_budget, seed=42):
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)
    pool_size = 12
    pool = []
    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n += 1
        if len(pool) >= 4 and rng.random() < 0.8:
            vals = np.array([p[0] for p in pool])
            probs = vals.max() - vals + 1e-8
            probs /= probs.sum()

            breed_type = rng.integers(0, 4)
            if breed_type == 0:
                idx1, idx2 = rng.choice(len(pool), 2, replace=False, p=probs)
                alpha = rng.beta(2, 2)
                x_init = alpha * pool[idx1][1] + (1 - alpha) * pool[idx2][1]
                x_init += rng.standard_cauchy(P) * 0.015
            elif breed_type == 1:
                idx = rng.choice(len(pool), 3, replace=False, p=probs)
                w = rng.dirichlet(np.ones(3))
                x_init = sum(w[i] * pool[idx[i]][1] for i in range(3))
                x_init += 0.2 * rng.standard_normal(P) * np.mean(x_init)
            elif breed_type == 2:
                best_pool_idx = np.argmin(vals)
                x_init = pool[best_pool_idx][1].copy()
                k = max(3, P // 8)
                idx_p = rng.choice(P, k, replace=False)
                x_init[idx_p] *= np.exp(0.5 * rng.standard_normal(k))
            else:
                idx = rng.choice(len(pool), p=probs)
                x_init = pool[idx][1] + 0.5 * rng.standard_normal(P) * np.mean(pool[idx][1])

            x_init = np.maximum(x_init, 0.0)
            if x_init.sum() < 1e-12:
                x_init = rng.dirichlet(np.ones(P))
            else:
                x_init /= x_init.sum()
        else:
            choice = rng.integers(0, 5)
            if choice == 0:
                x_init = rng.dirichlet(np.full(P, 0.05))
            elif choice == 1:
                z = np.abs(rng.standard_cauchy(P))
                x_init = z / z.sum()
            elif choice == 2:
                x_init = rng.random(P) ** rng.uniform(3, 12)
                x_init /= x_init.sum()
            elif choice == 3:
                x_init = np.exp(rng.uniform(1, 4) * rng.standard_normal(P))
                x_init /= x_init.sum()
            else:
                x_init = rng.dirichlet(np.full(P, 0.3))

        _, pol_v, x = _hybrid_single_restart_fast(
            x_init, P, beta_arr, 15000, 200000)
        if len(pool) < pool_size:
            pool.append((pol_v, x.copy()))
        else:
            worst = max(range(len(pool)), key=lambda i: pool[i][0])
            if pol_v < pool[worst][0]:
                pool[worst] = (pol_v, x.copy())
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# Optimized Method 5: Extreme Sparse Init
# Cold start — uses full BETA_ULTRA + fast core
# ============================================================================

def run_extreme_sparse_fast(time_budget, seed=42):
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)
    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n += 1
        k = rng.integers(3, 15)
        x_init = np.zeros(P)
        active_idx = rng.choice(P, k, replace=False)
        x_init[active_idx] = rng.dirichlet(np.ones(k))
        _, pol_v, x = _hybrid_single_restart_fast(
            x_init, P, beta_arr, 15000, 200000)
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# Registry
# ============================================================================

ALL_METHODS_FAST = {
    "iterated_warm": ("Iterated Warm Restart (Fast)", run_iterated_warm_fast),
    "mirrored_sampling": ("Mirrored Sampling (Fast)", run_mirrored_sampling_fast),
    "adaptive_perturbation": ("Adaptive Perturbation (Fast)", run_adaptive_perturbation_fast),
    "heavy_elite_v2": ("Heavy-Tail Elite v2 (Fast)", run_heavy_elite_v2_fast),
    "extreme_sparse": ("Extreme Sparse Init (Fast)", run_extreme_sparse_fast),
}
