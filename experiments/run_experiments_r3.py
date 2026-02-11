"""
Round 3 experiments: Combine winners + novel approaches.

Key findings from R1-R2:
- Heavy-tailed init (Cauchy/power-law) helps at P=200 (1.5117 vs baseline 1.5121)
- Warm cascade (low-P explore -> upsample) helps at P=50,100
- The standard LSE+Polyak pipeline is hard to beat, but init matters a lot

Round 3 ideas:
- Combine heavy-tail init + warm cascade
- Pool-based breeding of best solutions from heavy-tailed starts
- Solution recombination: extract "good patterns" from top solutions
- Alternating LSE/Polyak phases (interleaved, not sequential)
- Penalty reformulation: penalize peak variance instead of peak max
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidon_core import (
    autoconv_coeffs, project_simplex_nb, lse_obj_nb, lse_grad_nb,
    lse_objgrad_nb, armijo_step_nb_v2, _autoconv_max_argmax,
    _polyak_polish_nb, _hybrid_single_restart, _cyclic_polish_nb,
    make_inits, upsample_solution, BETA_HEAVY, logsumexp_nb, softmax_nb
)


def verify_solution(x, P):
    c = autoconv_coeffs(x, P)
    return float(np.max(c))


def save_solution(name, P, x, val):
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"solution_{name}_P{P}.json")
    data = {"method": name, "P": P, "value": val, "x": x.tolist()}
    with open(fname, "w") as f:
        json.dump(data, f)


# ============================================================================
# R3-1: Heavy-Tail Init + Warm Cascade (combine two winners)
# ============================================================================

def run_heavy_tail_cascade(P, time_budget):
    """
    Combine heavy-tailed initialization with warm-start cascade.
    Phase 1: Heavy-tailed exploration at low P
    Phase 2: Upsample best to target P + polish
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    P_low = max(20, P // 3)
    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    t_low_budget = time_budget * 0.35

    # Phase 1: heavy-tailed exploration at low P
    low_pool = []
    while time.time() - t0 < t_low_budget:
        choice = n_restarts % 4
        if choice == 0:
            x_init = rng.dirichlet(np.full(P_low, 0.05))
        elif choice == 1:
            z = np.abs(rng.standard_cauchy(P_low))
            x_init = z / z.sum()
        elif choice == 2:
            alpha = rng.uniform(3, 10)
            x_init = rng.random(P_low) ** alpha
            x_init /= x_init.sum()
        else:
            sigma = rng.uniform(1.0, 3.0)
            x_init = np.exp(sigma * rng.standard_normal(P_low))
            x_init /= x_init.sum()

        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P_low, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)
        n_restarts += 1
        low_pool.append((pol_v, x.copy()))

    low_pool.sort(key=lambda t: t[0])
    top_k = min(5, len(low_pool))

    # Phase 2: upsample + heavy-tail restarts at target P
    while time.time() - t0 < time_budget:
        if rng.random() < 0.6 and top_k > 0:
            # Upsample from pool
            idx = rng.integers(0, top_k)
            x_init = upsample_solution(low_pool[idx][1], P_low, P)
            noise_scale = rng.uniform(0.1, 0.4)
            x_init += noise_scale * rng.standard_normal(P) * np.mean(x_init)
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        else:
            # Fresh heavy-tailed init
            choice = rng.integers(0, 4)
            if choice == 0:
                x_init = rng.dirichlet(np.full(P, 0.05))
            elif choice == 1:
                z = np.abs(rng.standard_cauchy(P))
                x_init = z / z.sum()
            elif choice == 2:
                alpha = rng.uniform(3, 10)
                x_init = rng.random(P) ** alpha
                x_init /= x_init.sum()
            else:
                sigma = rng.uniform(1.0, 3.0)
                x_init = np.exp(sigma * rng.standard_normal(P))
                x_init /= x_init.sum()

        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)
        n_restarts += 1

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# R3-2: Elite Breeding with Heavy-Tail Mutations
# ============================================================================

def run_elite_breeding(P, time_budget):
    """
    Start with heavy-tailed inits. After building a pool of elite solutions,
    breed new candidates by:
    1. Interpolating between two parents
    2. Adding heavy-tailed mutation
    3. Running LSE+Polyak on the offspring
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    pool_size = 8

    pool = []
    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n_restarts += 1

        if len(pool) >= 3 and rng.random() < 0.6:
            # Breed from pool
            vals = np.array([p[0] for p in pool])
            # Fitness-proportional selection (lower = better)
            probs = vals.max() - vals + 1e-8
            probs /= probs.sum()
            idx1, idx2 = rng.choice(len(pool), 2, replace=False, p=probs)
            alpha = rng.beta(2, 2)  # favor middle
            x_init = alpha * pool[idx1][1] + (1 - alpha) * pool[idx2][1]

            # Heavy-tailed mutation
            mutation_type = rng.integers(0, 3)
            if mutation_type == 0:
                # Cauchy noise
                noise = rng.standard_cauchy(P) * 0.02
                x_init += noise
            elif mutation_type == 1:
                # Sparse perturbation
                k = max(2, P // 10)
                idx_perturb = rng.choice(P, k, replace=False)
                x_init[idx_perturb] *= rng.uniform(0.5, 2.0, k)
            else:
                # Gaussian noise
                x_init += 0.3 * rng.standard_normal(P) * np.mean(x_init)

            x_init = np.maximum(x_init, 0.0)
            if x_init.sum() < 1e-12:
                x_init = rng.dirichlet(np.ones(P))
            else:
                x_init /= x_init.sum()
        else:
            # Heavy-tailed fresh init
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

        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)

        if len(pool) < pool_size:
            pool.append((pol_v, x.copy()))
        else:
            worst_idx = max(range(len(pool)), key=lambda i: pool[i][0])
            if pol_v < pool[worst_idx][0]:
                pool[worst_idx] = (pol_v, x.copy())

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# R3-3: Interleaved LSE/Polyak (alternating phases)
# ============================================================================

def run_interleaved_lse_polyak(P, time_budget):
    """
    Instead of doing all LSE stages then all Polyak, alternate:
    LSE(beta_low) -> Polyak(short) -> LSE(beta_med) -> Polyak(short) -> ...
    The Polyak phases fix the argmax during LSE continuation.
    """
    rng = np.random.default_rng(42)
    beta_stages = [1, 3, 10, 30, 100, 300, 1000, 3000]

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P))
        n_restarts += 1

        for beta in beta_stages:
            beta_arr = np.array([beta], dtype=np.float64)
            # Short LSE phase
            _, _, x = _hybrid_single_restart(x, P, beta_arr, n_iters_lse=2000, n_iters_polyak=0)
            # Short Polyak polish
            pol_v, x = _polyak_polish_nb(x, P, 20000)

        # Final long Polyak
        pol_v, x = _polyak_polish_nb(x, P, 100000)

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# R3-4: Peak Variance Penalty
# ============================================================================

def run_peak_variance_penalty(P, time_budget):
    """
    Instead of just minimizing max(c_k), also penalize variance of top peaks.
    This encourages flatter peak profiles which are easier to minimize.
    Use LSE with a variance penalty term.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P))
        n_restarts += 1

        # Standard LSE warmup
        _, _, x = _hybrid_single_restart(x, P, beta_arr, n_iters_lse=10000, n_iters_polyak=0)

        # Peak variance reduction phase
        local_best = float(np.max(autoconv_coeffs(x, P)))
        local_best_x = x.copy()
        scale4P = 2.0 * (2.0 * P)

        for t in range(200000):
            c = autoconv_coeffs(x, P)
            fval = float(np.max(c))
            k_star = int(np.argmax(c))

            if fval < local_best:
                local_best = fval
                local_best_x = x.copy()

            # Find top-5 peaks
            top_k = min(5, len(c))
            top_idx = np.argpartition(c, -top_k)[-top_k:]
            top_vals = c[top_idx]
            mean_top = np.mean(top_vals)

            # Gradient: push down peaks above mean, ignore below mean
            g = np.zeros(P)
            n_above = 0
            for ki in top_idx:
                if c[ki] >= mean_top:
                    for i in range(P):
                        j = ki - i
                        if 0 <= j < P:
                            g[i] += scale4P * x[j]
                    n_above += 1
            if n_above > 0:
                g /= n_above

            gnorm2 = float(np.dot(g, g))
            if gnorm2 < 1e-20:
                break

            offset = 0.01 / (1.0 + t * 1e-4)
            target = local_best - offset
            step = (fval - target) / gnorm2
            if step < 0:
                step = 1e-5 / (1.0 + t * 1e-4)

            x = x - step * g
            x = project_simplex_nb(x)

        if local_best < best_val:
            best_val = local_best
            best_x = local_best_x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# R3-5: Multi-Seed Best-of-K with Diverse Seeds
# ============================================================================

def run_multi_seed_diverse(P, time_budget):
    """
    Instead of one RNG seed, use multiple diverse seeds and pick the best.
    Also mix initialization strategies systematically.
    """
    rng = np.random.default_rng()
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    strategies_cycle = [
        lambda rng, P: rng.dirichlet(np.full(P, 0.05)),  # very sparse
        lambda rng, P: np.abs(rng.standard_cauchy(P)) / np.abs(rng.standard_cauchy(P)).sum(),  # cauchy
        lambda rng, P: rng.dirichlet(np.ones(P)),  # uniform
        lambda rng, P: rng.dirichlet(np.full(P, 3.0)),  # concentrated
    ]

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Pick strategy
        strat = strategies_cycle[n_restarts % len(strategies_cycle)]
        x_init = strat(rng, P)

        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)
        n_restarts += 1

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# R3-6: Rescaled Gradient (adaptive learning rate per coordinate)
# ============================================================================

def run_rescaled_gradient(P, time_budget):
    """
    Use adaptive per-coordinate learning rates (like Adam but simpler).
    Coordinates with historically large gradients get smaller steps.
    This prevents the optimizer from over-correcting popular variables.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P))
        n_restarts += 1

        # LSE warmup
        _, _, x = _hybrid_single_restart(x, P, beta_arr, n_iters_lse=15000, n_iters_polyak=0)

        local_best = float(np.max(autoconv_coeffs(x, P)))
        local_best_x = x.copy()
        scale4P = 2.0 * (2.0 * P)

        # Adam-style state
        v = np.zeros(P)  # second moment
        eps = 1e-8

        for t in range(200000):
            c = autoconv_coeffs(x, P)
            fval = float(np.max(c))
            k_star = int(np.argmax(c))

            if fval < local_best:
                local_best = fval
                local_best_x = x.copy()

            # Subgradient
            g = np.zeros(P)
            for i in range(P):
                j = k_star - i
                if 0 <= j < P:
                    g[i] = scale4P * x[j]

            # Update second moment
            v = 0.999 * v + 0.001 * g * g

            # Rescaled step
            offset = 0.01 / (1.0 + t * 1e-4)
            target = local_best - offset
            eta = 0.001 / (1.0 + t * 1e-5)

            g_rescaled = g / (np.sqrt(v) + eps)
            x = x - eta * g_rescaled
            x = project_simplex_nb(x)

        if local_best < best_val:
            best_val = local_best
            best_x = local_best_x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# R3-7: Quadratic Approximation near Peak
# ============================================================================

def run_quadratic_approx(P, time_budget):
    """
    Near the optimum, approximate the max-peak objective locally as a
    quadratic and solve the quadratic subproblem. This uses the structure
    of the autoconvolution (c_k is quadratic in x).

    For the current argmax k*, c_{k*}(x) = 2P * sum_{i+j=k*} x_i*x_j.
    This IS a quadratic form in x. We can try to minimize it directly
    subject to simplex constraints.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P))
        n_restarts += 1

        # Full LSE+Polyak first
        lse_v, pol_v, x = _hybrid_single_restart(
            x, P, beta_arr, n_iters_lse=15000, n_iters_polyak=100000)

        # Quadratic refinement: for current argmax k*,
        # c_{k*}(x) = 2P * sum_{i+j=k*} x_i*x_j = x^T Q_{k*} x
        # where Q_{k*}[i,j] = 2P if i+j==k* else 0
        # Try projected gradient on this quadratic
        for outer in range(5):
            c = autoconv_coeffs(x, P)
            k_star = int(np.argmax(c))

            # Gradient of c_{k*}: g[i] = 2 * 2P * x[k*-i]
            scale4P = 2.0 * (2.0 * P)
            for inner in range(5000):
                g = np.zeros(P)
                for i in range(P):
                    j = k_star - i
                    if 0 <= j < P:
                        g[i] = scale4P * x[j]

                eta = 0.001 / (1.0 + inner * 0.01)
                x_new = x - eta * g
                x_new = project_simplex_nb(x_new)

                # Check if argmax changed
                c_new = autoconv_coeffs(x_new, P)
                if np.argmax(c_new) != k_star:
                    x = x_new
                    break
                x = x_new

        fval = float(np.max(autoconv_coeffs(x, P)))
        if fval < best_val:
            best_val = fval
            best_x = x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# R3-8: Restart from Best Known Solution + Perturbations
# ============================================================================

def run_warm_from_best(P, time_budget):
    """
    Load the best known solution from R2 and do warm restarts from it.
    This is like the "warm_perturb" strategy in the cloud runner.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    # Load best known solution
    best_known_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f"solution_heavy_tail_init_P{P}.json")
    if os.path.exists(best_known_file):
        with open(best_known_file) as f:
            data = json.load(f)
        warm_x = np.array(data['x'])
    else:
        warm_x = np.ones(P) / P

    best_val = verify_solution(warm_x, P)
    best_x = warm_x.copy()
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n_restarts += 1
        # Perturb warm start with varying noise levels
        noise_scale = rng.uniform(0.1, 1.5)
        x_init = warm_x + noise_scale * rng.standard_normal(P) * np.mean(warm_x)
        x_init = np.maximum(x_init, 0.0)
        if x_init.sum() < 1e-12:
            x_init = rng.dirichlet(np.ones(P))
        else:
            x_init /= x_init.sum()

        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
            warm_x = x.copy()  # Update warm start

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Main runner
# ============================================================================

ALL_EXPERIMENTS_R3 = {
    "heavy_tail_cascade": ("Heavy-Tail + Warm Cascade", run_heavy_tail_cascade),
    "elite_breeding": ("Elite Breeding + Heavy-Tail", run_elite_breeding),
    "interleaved_lse_polyak": ("Interleaved LSE/Polyak", run_interleaved_lse_polyak),
    "peak_variance_penalty": ("Peak Variance Penalty", run_peak_variance_penalty),
    "multi_seed_diverse": ("Multi-Seed Diverse Cycling", run_multi_seed_diverse),
    "rescaled_gradient": ("Rescaled Gradient (Adam-like)", run_rescaled_gradient),
    "quadratic_approx": ("Quadratic Approximation", run_quadratic_approx),
    "warm_from_best": ("Warm Restart from Best Known", run_warm_from_best),
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--P", type=str, default="50,100,200")
    parser.add_argument("--methods", type=str, default="all")
    args = parser.parse_args()

    P_values = tuple(int(p) for p in args.P.split(","))

    # Warmup
    print("Warming up Numba JIT...")
    x_test = np.ones(5) / 5.0
    _ = autoconv_coeffs(x_test, 5)
    _ = _polyak_polish_nb(x_test, 5, 10)
    _ = _cyclic_polish_nb(x_test, 5, 10)
    beta_arr = np.array([1.0, 10.0])
    _ = _hybrid_single_restart(x_test, 5, beta_arr, 10, 10)
    print("Warmup complete.\n")

    if args.methods == "all":
        method_keys = list(ALL_EXPERIMENTS_R3.keys())
    else:
        method_keys = args.methods.split(",")

    print(f"Running methods: {method_keys}")
    print(f"Time budget: {args.budget}s per method per P value")
    print(f"P values: {P_values}")

    results = {}
    for key in method_keys:
        name, func = ALL_EXPERIMENTS_R3[key]
        results[key] = {}
        print(f"\n{'-' * 60}")
        print(f"Method: {name}")
        print(f"{'-' * 60}")
        for P in P_values:
            print(f"  P={P}...", end=" ", flush=True)
            val, x, restarts, elapsed = func(P, args.budget)
            results[key][P] = {"value": val, "restarts": restarts, "elapsed": elapsed}
            save_solution(key, P, x, val)
            print(f"val={val:.6f}  restarts={restarts}  time={elapsed:.1f}s")

    # Summary
    print(f"\n\n{'=' * 80}")
    print("ROUND 3 SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Method':<40s} {'P=50':>10s} {'P=100':>10s} {'P=200':>10s}")
    print(f"{'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"{'Baseline (LSE+Polyak)':<40s} {'1.519790':>10s} {'1.515085':>10s} {'1.512053':>10s}")
    print(f"{'Heavy-Tail Init (R2 winner)':<40s} {'1.519322':>10s} {'1.514898':>10s} {'1.511674':>10s}")

    for key in results:
        name = ALL_EXPERIMENTS_R3[key][0]
        vals = []
        for P in P_values:
            if P in results[key]:
                vals.append(f"{results[key][P]['value']:.6f}")
            else:
                vals.append("N/A")
        print(f"{name:<40s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_data_r3.json")
    save_results = {}
    for key in results:
        save_results[key] = {}
        for P in results[key]:
            r = results[key][P]
            save_results[key][str(P)] = {"value": r["value"], "restarts": r["restarts"], "elapsed": r["elapsed"]}
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
