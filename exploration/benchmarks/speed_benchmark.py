"""
Baseline speed benchmark for 5 CMA-ES-style Sidon optimization methods.

Times individual components and full method runs at P=200 with 30s budget.
"""

import sys, os, time, json
import numpy as np

# Setup paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")

from sidon_core import (
    autoconv_coeffs, project_simplex_nb, lse_obj_nb, lse_grad_nb,
    lse_objgrad_nb, armijo_step_nb_v2, _autoconv_max_argmax,
    _polyak_polish_nb, _hybrid_single_restart, _cyclic_polish_nb,
    BETA_HEAVY, BETA_ULTRA, logsumexp_nb, softmax_nb
)

P = 200
TIME_BUDGET = 30  # seconds per method


# ============================================================================
# Helpers
# ============================================================================

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
# Component benchmarks
# ============================================================================

def bench_autoconv(n_calls=10000):
    """Time autoconv_coeffs at P=200."""
    x = np.random.default_rng(42).dirichlet(np.ones(P))
    t0 = time.perf_counter()
    for _ in range(n_calls):
        c = autoconv_coeffs(x, P)
    elapsed = time.perf_counter() - t0
    return elapsed / n_calls


def bench_autoconv_max_argmax(n_calls=10000):
    """Time _autoconv_max_argmax at P=200."""
    x = np.random.default_rng(42).dirichlet(np.ones(P))
    t0 = time.perf_counter()
    for _ in range(n_calls):
        c_max, k_max = _autoconv_max_argmax(x, P)
    elapsed = time.perf_counter() - t0
    return elapsed / n_calls


def bench_lse_objgrad(n_calls=5000):
    """Time lse_objgrad_nb at P=200."""
    x = np.random.default_rng(42).dirichlet(np.ones(P))
    beta = 100.0
    t0 = time.perf_counter()
    for _ in range(n_calls):
        obj, g = lse_objgrad_nb(x, P, beta)
    elapsed = time.perf_counter() - t0
    return elapsed / n_calls


def bench_project_simplex(n_calls=10000):
    """Time project_simplex_nb at P=200."""
    x = np.random.default_rng(42).standard_normal(P)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        y = project_simplex_nb(x)
    elapsed = time.perf_counter() - t0
    return elapsed / n_calls


def bench_polyak(n_iters=50000):
    """Time _polyak_polish_nb for n_iters iterations."""
    x = np.random.default_rng(42).dirichlet(np.ones(P))
    t0 = time.perf_counter()
    val, x_out = _polyak_polish_nb(x, P, n_iters)
    elapsed = time.perf_counter() - t0
    return elapsed, n_iters / elapsed, val


def bench_hybrid_single(n_iters_lse=500, n_iters_polyak=10000):
    """Time a single _hybrid_single_restart call."""
    x = np.random.default_rng(42).dirichlet(np.ones(P))
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    t0 = time.perf_counter()
    lse_v, pol_v, x_out = _hybrid_single_restart(x, P, beta_arr, n_iters_lse, n_iters_polyak)
    elapsed = time.perf_counter() - t0
    return elapsed, pol_v


# ============================================================================
# 5 methods (copied from run_experiments_s2.py and run_push_below.py)
# ============================================================================

def run_iterated_warm(time_budget, seed=42):
    """Iterated Warm Restart: perturb best -> optimize -> repeat."""
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)
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
        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
            warm_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


def run_mirrored_sampling(time_budget, seed=42):
    """Mirrored Sampling: try x+z and x-z for each perturbation z."""
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
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

        _, pol_v1, x1 = _hybrid_single_restart(x_plus, P, beta_arr, 15000, 200000)
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

        _, pol_v2, x2 = _hybrid_single_restart(x_minus, P, beta_arr, 15000, 200000)
        n += 1
        if pol_v2 < best_val:
            best_val = pol_v2
            best_x = x2.copy()
            warm_x = x2.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


def run_adaptive_perturbation(time_budget, seed=42):
    """Adaptive Perturbation: Thompson sampling over perturbation scales."""
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
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

        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
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


def run_heavy_elite_v2(time_budget, seed=42):
    """Heavy-Tail Elite v2: pool-based breeding with heavy-tailed inits."""
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

        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
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


def run_extreme_sparse(time_budget, seed=42):
    """Extreme Sparse Init: very sparse inits (3-15 active bins)."""
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
        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# Main benchmark runner
# ============================================================================

ALL_METHODS = {
    "iterated_warm": ("Iterated Warm Restart", run_iterated_warm),
    "mirrored_sampling": ("Mirrored Sampling", run_mirrored_sampling),
    "adaptive_perturbation": ("Adaptive Perturbation", run_adaptive_perturbation),
    "heavy_elite_v2": ("Heavy-Tail Elite v2", run_heavy_elite_v2),
    "extreme_sparse": ("Extreme Sparse Init", run_extreme_sparse),
}


def run_component_benchmarks():
    """Benchmark individual computational components."""
    print("=" * 70)
    print("COMPONENT BENCHMARKS (P=200)")
    print("=" * 70)

    t_autoconv = bench_autoconv(10000)
    print(f"  autoconv_coeffs:       {t_autoconv*1e6:8.1f} us/call  ({1/t_autoconv:.0f} calls/s)")

    t_maxarg = bench_autoconv_max_argmax(10000)
    print(f"  _autoconv_max_argmax:  {t_maxarg*1e6:8.1f} us/call  ({1/t_maxarg:.0f} calls/s)")

    t_lse = bench_lse_objgrad(5000)
    print(f"  lse_objgrad_nb:        {t_lse*1e6:8.1f} us/call  ({1/t_lse:.0f} calls/s)")

    t_proj = bench_project_simplex(10000)
    print(f"  project_simplex_nb:    {t_proj*1e6:8.1f} us/call  ({1/t_proj:.0f} calls/s)")

    pol_time, pol_rate, pol_val = bench_polyak(50000)
    print(f"  _polyak_polish (50K):  {pol_time:8.2f} s  ({pol_rate:.0f} iters/s, val={pol_val:.6f})")

    hyb_time, hyb_val = bench_hybrid_single(500, 10000)
    print(f"  _hybrid_single (500+10K): {hyb_time:6.2f} s  (val={hyb_val:.6f})")

    return {
        "autoconv_us": t_autoconv * 1e6,
        "autoconv_max_argmax_us": t_maxarg * 1e6,
        "lse_objgrad_us": t_lse * 1e6,
        "project_simplex_us": t_proj * 1e6,
        "polyak_50k_s": pol_time,
        "polyak_iters_per_s": pol_rate,
        "hybrid_500_10k_s": hyb_time,
    }


def run_method_benchmarks():
    """Run each of the 5 methods for TIME_BUDGET seconds."""
    print(f"\n{'=' * 70}")
    print(f"METHOD BENCHMARKS (P={P}, budget={TIME_BUDGET}s each)")
    print(f"{'=' * 70}")

    results = {}
    for key, (name, func) in ALL_METHODS.items():
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


if __name__ == "__main__":
    print("Warming up Numba JIT...")
    x_test = np.ones(5) / 5.0
    _ = autoconv_coeffs(x_test, 5)
    _ = _autoconv_max_argmax(x_test, 5)
    _ = _polyak_polish_nb(x_test, 5, 10)
    _ = _cyclic_polish_nb(x_test, 5, 10)
    beta_arr = np.array([1.0, 10.0])
    _ = _hybrid_single_restart(x_test, 5, beta_arr, 10, 10)
    _ = lse_objgrad_nb(x_test, 5, 10.0)
    _ = project_simplex_nb(x_test)
    print("Warmup complete.\n")

    component_results = run_component_benchmarks()
    method_results = run_method_benchmarks()

    # Summary
    print(f"\n\n{'=' * 70}")
    print("BASELINE SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Method':<30s} {'Value':>10s} {'Restarts':>10s} {'Rate':>12s}")
    print(f"{'-' * 30} {'-' * 10} {'-' * 10} {'-' * 12}")
    for key in ALL_METHODS:
        r = method_results[key]
        print(f"{r['name']:<30s} {r['value']:10.6f} {r['restarts']:10d} "
              f"{r['restarts_per_s']:10.3f}/s")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "benchmark_baseline.json")
    save_data = {
        "P": P,
        "time_budget": TIME_BUDGET,
        "components": component_results,
        "methods": method_results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {out_path}")
