"""
Final comparison: top 5 methods from R1-R3, longer budget for better statistics.
Run each method 3 times with different seeds to check robustness.
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidon_core import (
    autoconv_coeffs, project_simplex_nb,
    lse_objgrad_nb, armijo_step_nb_v2,
    _polyak_polish_nb, _hybrid_single_restart,
    make_inits, upsample_solution, BETA_HEAVY
)


def verify_solution(x, P):
    c = autoconv_coeffs(x, P)
    return float(np.max(c))


def save_solution(name, P, x, val):
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"solution_final_{name}_P{P}.json")
    data = {"method": name, "P": P, "value": val, "x": x.tolist()}
    with open(fname, "w") as f:
        json.dump(data, f)


# ============================================================================
# Method implementations (condensed from R1-R3)
# ============================================================================

def baseline_lse_polyak(P, time_budget, seed=42):
    """Standard LSE+Polyak with Dirichlet init."""
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    best_val = np.inf
    best_x = None
    n = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        x_init = rng.dirichlet(np.ones(P))
        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
    return verify_solution(best_x, P), best_x, n, time.time() - t0


def heavy_tail_init(P, time_budget, seed=42):
    """Heavy-tailed initialization (Cauchy, power-law, log-normal)."""
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    best_val = np.inf
    best_x = None
    n = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        choice = n % 4
        if choice == 0:
            x_init = rng.dirichlet(np.full(P, 0.05))
        elif choice == 1:
            z = np.abs(rng.standard_cauchy(P))
            x_init = z / z.sum()
        elif choice == 2:
            x_init = rng.random(P) ** rng.uniform(3, 10)
            x_init /= x_init.sum()
        else:
            x_init = np.exp(rng.uniform(1, 3) * rng.standard_normal(P))
            x_init /= x_init.sum()
        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
    return verify_solution(best_x, P), best_x, n, time.time() - t0


def elite_breeding(P, time_budget, seed=42):
    """Pool-based breeding with heavy-tailed mutation."""
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    pool_size = 8
    pool = []
    best_val = np.inf
    best_x = None
    n = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        n += 1
        if len(pool) >= 3 and rng.random() < 0.6:
            vals = np.array([p[0] for p in pool])
            probs = vals.max() - vals + 1e-8
            probs /= probs.sum()
            idx1, idx2 = rng.choice(len(pool), 2, replace=False, p=probs)
            alpha = rng.beta(2, 2)
            x_init = alpha * pool[idx1][1] + (1 - alpha) * pool[idx2][1]
            mt = rng.integers(0, 3)
            if mt == 0:
                x_init += rng.standard_cauchy(P) * 0.02
            elif mt == 1:
                k = max(2, P // 10)
                idx_p = rng.choice(P, k, replace=False)
                x_init[idx_p] *= rng.uniform(0.5, 2.0, k)
            else:
                x_init += 0.3 * rng.standard_normal(P) * np.mean(x_init)
            x_init = np.maximum(x_init, 0.0)
            if x_init.sum() < 1e-12:
                x_init = rng.dirichlet(np.ones(P))
            else:
                x_init /= x_init.sum()
        else:
            choice = rng.integers(0, 4)
            if choice == 0:
                x_init = rng.dirichlet(np.full(P, 0.05))
            elif choice == 1:
                z = np.abs(rng.standard_cauchy(P))
                x_init = z / z.sum()
            elif choice == 2:
                x_init = rng.random(P) ** rng.uniform(3, 10)
                x_init /= x_init.sum()
            else:
                x_init = np.exp(rng.uniform(1, 3) * rng.standard_normal(P))
                x_init /= x_init.sum()
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
    return verify_solution(best_x, P), best_x, n, time.time() - t0


def interleaved_lse_polyak(P, time_budget, seed=42):
    """Alternating LSE and Polyak phases."""
    rng = np.random.default_rng(seed)
    beta_stages = [1, 3, 10, 30, 100, 300, 1000, 3000]
    best_val = np.inf
    best_x = None
    n = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P))
        n += 1
        for beta in beta_stages:
            beta_arr = np.array([beta], dtype=np.float64)
            _, _, x = _hybrid_single_restart(x, P, beta_arr, 2000, 0)
            _, x = _polyak_polish_nb(x, P, 20000)
        pol_v, x = _polyak_polish_nb(x, P, 100000)
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
    return verify_solution(best_x, P), best_x, n, time.time() - t0


def warm_cascade(P, time_budget, seed=42):
    """Low-P exploration then upsample cascade."""
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    P_low = max(20, P // 3)
    best_val = np.inf
    best_x = None
    n = 0
    t0 = time.time()
    t_low = time_budget * 0.4
    low_pool = []
    while time.time() - t0 < t_low:
        x_init = rng.dirichlet(np.ones(P_low))
        _, pol_v, x = _hybrid_single_restart(x_init, P_low, beta_arr, 15000, 200000)
        n += 1
        low_pool.append((pol_v, x.copy()))
    low_pool.sort(key=lambda t: t[0])
    top_k = min(5, len(low_pool))
    while time.time() - t0 < time_budget:
        idx = rng.integers(0, top_k)
        x_up = upsample_solution(low_pool[idx][1], P_low, P)
        x_up += 0.2 * rng.standard_normal(P) * np.mean(x_up)
        x_up = np.maximum(x_up, 0.0)
        x_up /= x_up.sum()
        _, pol_v, x = _hybrid_single_restart(x_up, P, beta_arr, 15000, 200000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
    return verify_solution(best_x, P), best_x, n, time.time() - t0


# ============================================================================
# Main
# ============================================================================

METHODS = {
    "baseline": ("Baseline LSE+Polyak", baseline_lse_polyak),
    "heavy_tail": ("Heavy-Tail Init", heavy_tail_init),
    "elite_breed": ("Elite Breeding", elite_breeding),
    "interleaved": ("Interleaved LSE/Polyak", interleaved_lse_polyak),
    "warm_cascade": ("Warm Cascade", warm_cascade),
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=90)
    parser.add_argument("--P", type=str, default="50,100,200")
    parser.add_argument("--n_trials", type=int, default=3)
    args = parser.parse_args()

    P_values = tuple(int(p) for p in args.P.split(","))

    # Warmup
    print("Warming up Numba JIT...")
    x_test = np.ones(5) / 5.0
    _ = autoconv_coeffs(x_test, 5)
    _ = _polyak_polish_nb(x_test, 5, 10)
    beta_arr = np.array([1.0, 10.0])
    _ = _hybrid_single_restart(x_test, 5, beta_arr, 10, 10)
    print("Warmup complete.\n")

    all_results = {}
    for key, (name, func) in METHODS.items():
        all_results[key] = {}
        print(f"\n{'=' * 60}")
        print(f"Method: {name} ({args.n_trials} trials, {args.budget}s budget)")
        print(f"{'=' * 60}")

        for P in P_values:
            trials = []
            for trial in range(args.n_trials):
                seed = 42 + trial * 1000
                print(f"  P={P}, trial {trial+1}/{args.n_trials}...", end=" ", flush=True)
                val, x, restarts, elapsed = func(P, args.budget, seed=seed)
                trials.append({"value": val, "restarts": restarts, "elapsed": elapsed})
                print(f"val={val:.6f}  restarts={restarts}")

                # Save best trial
                if trial == 0 or val < min(t["value"] for t in trials[:-1]):
                    save_solution(f"final_{key}", P, x, val)

            vals = [t["value"] for t in trials]
            all_results[key][str(P)] = {
                "best": min(vals),
                "mean": np.mean(vals),
                "std": np.std(vals),
                "trials": trials
            }
            print(f"  P={P}: best={min(vals):.6f}  mean={np.mean(vals):.6f}  std={np.std(vals):.6f}")

    # Final table
    print(f"\n\n{'=' * 90}")
    print("FINAL COMPARISON TABLE (best of 3 trials, 90s budget)")
    print(f"{'=' * 90}")
    print(f"{'Method':<30s} {'P=50 best':>12s} {'P=100 best':>12s} {'P=200 best':>12s} {'P=50 mean':>12s} {'P=100 mean':>12s} {'P=200 mean':>12s}")
    print(f"{'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

    for key in METHODS:
        name = METHODS[key][0]
        row = []
        for P in P_values:
            k = str(P)
            row.append(f"{all_results[key][k]['best']:.6f}")
        for P in P_values:
            k = str(P)
            row.append(f"{all_results[key][k]['mean']:.6f}")
        print(f"{name:<30s} {row[0]:>12s} {row[1]:>12s} {row[2]:>12s} {row[3]:>12s} {row[4]:>12s} {row[5]:>12s}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
