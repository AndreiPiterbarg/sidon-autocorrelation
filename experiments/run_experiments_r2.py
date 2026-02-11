"""
Round 2 experiments: More creative approaches that try to beat the baseline.

Key insight from Round 1: methods that don't use the powerful LSE+Polyak pipeline
as a building block can't compete. So Round 2 focuses on:
1. Improving the initialization fed into LSE+Polyak
2. Improving the Polyak phase itself
3. Hybrid population + gradient methods
4. Post-processing / solution refinement tricks
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidon_core import (
    autoconv_coeffs, project_simplex_nb, lse_obj_nb, lse_grad_nb,
    lse_objgrad_nb, armijo_step_nb_v2, _autoconv_max_argmax,
    _polyak_polish_nb, _hybrid_single_restart, _cyclic_polish_nb,
    make_inits, BETA_HEAVY, logsumexp_nb, softmax_nb
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
# Experiment R2-1: Elite Pool with Cross-Pollination
# ============================================================================

def run_elite_pool(P, time_budget):
    """
    Maintain an elite pool of K best solutions. After each LSE+Polyak restart,
    add to pool if good enough. Periodically breed new solutions by averaging
    pairs from the pool + noise.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    pool_size = 10

    pool = []  # (value, x)
    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Decide init strategy
        if len(pool) >= 3 and rng.random() < 0.5:
            # Breed from pool: pick 2 parents, interpolate + noise
            vals = [p[0] for p in pool]
            probs = np.array(vals)
            probs = probs.max() - probs + 1e-8  # lower val = higher prob
            probs /= probs.sum()
            idx1, idx2 = rng.choice(len(pool), 2, replace=False, p=probs)
            alpha = rng.uniform(0.3, 0.7)
            x_init = alpha * pool[idx1][1] + (1 - alpha) * pool[idx2][1]
            noise_scale = rng.uniform(0.1, 0.5)
            x_init += noise_scale * rng.standard_normal(P) * np.mean(x_init)
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        else:
            # Random init
            x_init = rng.dirichlet(np.ones(P))

        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)
        n_restarts += 1

        # Add to pool
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
# Experiment R2-2: Diverse Initialization Tournament
# ============================================================================

def run_diverse_init_tournament(P, time_budget):
    """
    Use ALL initialization strategies from sidon_core, not just Dirichlet.
    Run each with equal time share, keep the best.
    """
    strategies = [
        'dirichlet_uniform', 'dirichlet_sparse', 'dirichlet_concentrated',
        'gaussian_peak', 'bimodal', 'cosine_shaped', 'triangle',
        'flat_noisy', 'boundary_heavy', 'random_sparse_k',
        'symmetric_dirichlet',
    ]
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0
    time_per_strat = time_budget / len(strategies)

    t0 = time.time()
    for strat in strategies:
        strat_start = time.time()
        inits = make_inits(strat, P, 100, rng)
        idx = 0
        while time.time() - strat_start < time_per_strat and idx < len(inits):
            if time.time() - t0 >= time_budget:
                break
            x_init = inits[idx]
            idx += 1
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
# Experiment R2-3: Accelerated Polyak (Nesterov + Polyak hybrid)
# ============================================================================

def run_accelerated_polyak(P, time_budget):
    """
    After LSE warmup, use Polyak subgradient with Nesterov-style momentum.
    Standard Polyak doesn't use momentum; adding it might help escape traps.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x_init = rng.dirichlet(np.ones(P))
        n_restarts += 1

        # LSE phase (same as baseline)
        lse_v, _, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=0)

        # Accelerated Polyak phase
        x_prev = x.copy()
        local_best = float(np.max(autoconv_coeffs(x, P)))
        local_best_x = x.copy()
        scale4P = 2.0 * (2.0 * P)

        for t in range(200000):
            # Momentum extrapolation
            momentum = min(t / (t + 3.0), 0.9)
            y = x + momentum * (x - x_prev)
            y = project_simplex_nb(y)

            c = autoconv_coeffs(y, P)
            fval = float(np.max(c))
            k_star = int(np.argmax(c))

            if fval < local_best:
                local_best = fval
                local_best_x = y.copy()

            # Subgradient at y
            g = np.zeros(P)
            gnorm2 = 0.0
            for i in range(P):
                j = k_star - i
                if 0 <= j < P:
                    gi = scale4P * y[j]
                    g[i] = gi
                    gnorm2 += gi * gi

            if gnorm2 < 1e-20:
                break

            offset = 0.01 / (1.0 + t * 1e-4)
            target = local_best - offset
            step = (fval - target) / gnorm2
            if step < 0:
                step = 1e-5 / (1.0 + t * 1e-4)

            x_new = y - step * g
            x_new = project_simplex_nb(x_new)

            x_prev = x.copy()
            x = x_new

        if local_best < best_val:
            best_val = local_best
            best_x = local_best_x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment R2-4: LSE + Cyclic Peak Polish (from sidon_core)
# ============================================================================

def run_lse_cyclic_polish(P, time_budget):
    """
    Replace Polyak polish with cyclic peak-cutting polish.
    Cyclic targets near-peak indices in round-robin instead of always the argmax.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x_init = rng.dirichlet(np.ones(P))
        n_restarts += 1

        # LSE phase only
        lse_v, _, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=0)

        # Cyclic polish
        pol_val, pol_x = _cyclic_polish_nb(x, P, 200000)

        if pol_val < best_val:
            best_val = pol_val
            best_x = pol_x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment R2-5: Short LSE + Long Polyak (rebalance compute)
# ============================================================================

def run_short_lse_long_polyak(P, time_budget):
    """
    Instead of 15000 LSE iters + 200K Polyak, try 3000 LSE + 500K Polyak.
    Maybe we can get more restarts if LSE converges faster?
    Also try fewer beta stages.
    """
    rng = np.random.default_rng(42)
    # Shorter beta schedule (skip early stages)
    beta_short = np.array([1, 3, 10, 30, 100, 300, 1000, 3000], dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x_init = rng.dirichlet(np.ones(P))
        n_restarts += 1

        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_short, n_iters_lse=3000, n_iters_polyak=500000)

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment R2-6: Double Polyak Polish (two Polyak passes)
# ============================================================================

def run_double_polyak(P, time_budget):
    """
    After standard hybrid (LSE+Polyak), add perturbation and a second
    Polyak polish pass. The perturbation might escape the current basin.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x_init = rng.dirichlet(np.ones(P))
        n_restarts += 1

        # Standard hybrid
        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=100000)

        # Perturb + second polish
        x_perturbed = x + 0.1 * rng.standard_normal(P) * np.mean(x)
        x_perturbed = np.maximum(x_perturbed, 0.0)
        x_perturbed /= x_perturbed.sum()

        pol2_val, pol2_x = _polyak_polish_nb(x_perturbed, P, 100000)

        final_val = min(pol_v, pol2_val)
        final_x = pol2_x if pol2_val <= pol_v else x

        if final_val < best_val:
            best_val = final_val
            best_x = final_x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment R2-7: Warm Restart Cascade
# ============================================================================

def run_warm_cascade(P, time_budget):
    """
    Run the full budget as a warm-restart cascade:
    Start at P_low, optimize, upsample to P, re-optimize.
    This is the curriculum idea but within a single time budget.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    # Spend 40% of time at low P, 60% at target P
    if P <= 50:
        P_low = max(10, P // 2)
    else:
        P_low = max(20, P // 3)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    t_low_budget = time_budget * 0.4

    # Phase 1: explore at low P
    low_pool = []  # (val, x) at P_low
    while time.time() - t0 < t_low_budget:
        x_init = rng.dirichlet(np.ones(P_low))
        n_restarts += 1
        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P_low, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)
        low_pool.append((pol_v, x.copy()))

    # Sort and keep top 5
    low_pool.sort(key=lambda t: t[0])
    top_k = min(5, len(low_pool))

    # Phase 2: upsample best solutions and polish at target P
    from sidon_core import upsample_solution

    while time.time() - t0 < time_budget:
        # Pick from top solutions (with some randomization)
        idx = rng.integers(0, top_k)
        x_low = low_pool[idx][1]

        x_up = upsample_solution(x_low, P_low, P)
        # Add noise
        x_up += 0.2 * rng.standard_normal(P) * np.mean(x_up)
        x_up = np.maximum(x_up, 0.0)
        x_up /= x_up.sum()

        lse_v, pol_v, x = _hybrid_single_restart(
            x_up, P, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)
        n_restarts += 1

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment R2-8: Softmax Temperature Gradient (no LSE)
# ============================================================================

def run_softmax_temp_gradient(P, time_budget):
    """
    Instead of LSE continuation, directly minimize max_k c_k using
    gradient of c_{k*} where k* = argmax, but with a twist:
    use a softmax-weighted combination of ALL peak gradients as a search
    direction. Temperature parameter controls how soft/hard the weighting is.
    """
    rng = np.random.default_rng(42)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P) * 2.0)
        n_restarts += 1

        local_best = np.inf
        local_best_x = x.copy()

        # Temperature schedule (high -> low = soft -> hard)
        for temp in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
            for t in range(10000):
                if time.time() - t0 >= time_budget:
                    break

                c = autoconv_coeffs(x, P)
                fval = float(np.max(c))

                if fval < local_best:
                    local_best = fval
                    local_best_x = x.copy()

                # Softmax weights over peaks
                w = softmax_nb(c, 1.0 / temp)

                # Weighted gradient: g[i] = sum_k w_k * dc_k/dx_i
                g = np.zeros(P)
                for i in range(P):
                    s = 0.0
                    for j in range(P):
                        k = i + j
                        s += w[k] * x[j]
                    g[i] = 2.0 * (2.0 * P) * s

                gnorm = np.sqrt(np.dot(g, g))
                if gnorm < 1e-15:
                    break

                eta = 0.1 / (1.0 + t * 0.001) / gnorm * np.sqrt(P)
                x = x - eta * g
                x = project_simplex_nb(x)

        # Polyak polish
        pol_val, pol_x = _polyak_polish_nb(local_best_x, P, 200000)
        final_val = min(local_best, pol_val)
        final_x = pol_x if pol_val <= local_best else local_best_x

        if final_val < best_val:
            best_val = final_val
            best_x = final_x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment R2-9: Alternating Peak Targeting
# ============================================================================

def run_alternating_peaks(P, time_budget):
    """
    After LSE warmup, alternate between targeting the top-2 peaks explicitly.
    In each iteration, check which of the top-2 peaks is higher and cut it.
    This prevents the oscillation trap where cutting one peak raises another.

    Key idea: track the TOP TWO peaks and distribute gradient steps between them
    proportionally to how far each is above the target.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x_init = rng.dirichlet(np.ones(P))
        n_restarts += 1

        # LSE warmup
        lse_v, _, x = _hybrid_single_restart(
            x_init, P, beta_arr, n_iters_lse=15000, n_iters_polyak=0)

        local_best = float(np.max(autoconv_coeffs(x, P)))
        local_best_x = x.copy()
        scale4P = 2.0 * (2.0 * P)

        for t in range(200000):
            c = autoconv_coeffs(x, P)

            # Find top-2 peaks
            sorted_idx = np.argsort(c)[::-1]
            k1 = sorted_idx[0]
            k2 = sorted_idx[1]
            c1 = c[k1]
            c2 = c[k2]
            fval = c1

            if fval < local_best:
                local_best = fval
                local_best_x = x.copy()

            # Weighted gradient: combine gradients of top-2 peaks
            # Weight by how much each exceeds the target
            offset = 0.01 / (1.0 + t * 1e-4)
            target = local_best - offset

            w1 = max(c1 - target, 0.0)
            w2 = max(c2 - target, 0.0)
            wtotal = w1 + w2
            if wtotal < 1e-20:
                break
            w1 /= wtotal
            w2 /= wtotal

            g = np.zeros(P)
            gnorm2 = 0.0
            for i in range(P):
                gi = 0.0
                # Gradient from peak k1
                j1 = k1 - i
                if 0 <= j1 < P:
                    gi += w1 * scale4P * x[j1]
                # Gradient from peak k2
                j2 = k2 - i
                if 0 <= j2 < P:
                    gi += w2 * scale4P * x[j2]
                g[i] = gi
                gnorm2 += gi * gi

            if gnorm2 < 1e-20:
                break

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
# Experiment R2-10: Heavy-Tailed Restarts (Cauchy init)
# ============================================================================

def run_heavy_tail_init(P, time_budget):
    """
    Use heavy-tailed (Cauchy/Student-t) initialization instead of Dirichlet.
    This creates sparser initial solutions that explore different basins
    than the uniform Dirichlet.
    """
    rng = np.random.default_rng(42)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n_restarts += 1

        # Mix of initialization strategies
        choice = n_restarts % 4
        if choice == 0:
            # Very sparse Dirichlet
            x = rng.dirichlet(np.full(P, 0.05))
        elif choice == 1:
            # Cauchy-like: 1/pi * 1/(1+x^2), take absolute value
            z = rng.standard_cauchy(P)
            x = np.abs(z)
            x /= x.sum()
        elif choice == 2:
            # Power-law: x_i ~ U(0,1)^alpha for large alpha (concentrates on few)
            alpha = rng.uniform(3, 10)
            x = rng.random(P) ** alpha
            x /= x.sum()
        else:
            # Log-normal: exp(N(0, sigma^2))
            sigma = rng.uniform(1.0, 3.0)
            x = np.exp(sigma * rng.standard_normal(P))
            x /= x.sum()

        lse_v, pol_v, x_out = _hybrid_single_restart(
            x, P, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)

        if pol_v < best_val:
            best_val = pol_v
            best_x = x_out.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment R2-11: Aggressive Beta Schedule (faster continuation)
# ============================================================================

def run_aggressive_beta(P, time_budget):
    """
    Use a much more aggressive beta schedule (exponential growth)
    so we spend less time in the smooth regime and more in the sharp regime.
    Trade LSE quality for more restarts.
    """
    rng = np.random.default_rng(42)
    # Aggressive: fewer stages, bigger jumps, higher max
    beta_aggressive = np.array([1, 5, 25, 125, 625, 3000, 10000], dtype=np.float64)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x_init = rng.dirichlet(np.ones(P))
        n_restarts += 1

        lse_v, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_aggressive, n_iters_lse=5000, n_iters_polyak=300000)

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Main runner
# ============================================================================

ALL_EXPERIMENTS_R2 = {
    "elite_pool": ("Elite Pool + Cross-Pollination", run_elite_pool),
    "diverse_init": ("Diverse Init Tournament", run_diverse_init_tournament),
    "accelerated_polyak": ("Accelerated Polyak (Nesterov)", run_accelerated_polyak),
    "lse_cyclic": ("LSE + Cyclic Polish", run_lse_cyclic_polish),
    "short_lse_long_polyak": ("Short LSE + Long Polyak", run_short_lse_long_polyak),
    "double_polyak": ("Double Polyak Polish", run_double_polyak),
    "warm_cascade": ("Warm Restart Cascade", run_warm_cascade),
    "softmax_temp": ("Softmax Temperature Gradient", run_softmax_temp_gradient),
    "alternating_peaks": ("Alternating Peak Targeting", run_alternating_peaks),
    "heavy_tail_init": ("Heavy-Tailed Initialization", run_heavy_tail_init),
    "aggressive_beta": ("Aggressive Beta Schedule", run_aggressive_beta),
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
        method_keys = list(ALL_EXPERIMENTS_R2.keys())
    else:
        method_keys = args.methods.split(",")

    print(f"Running methods: {method_keys}")
    print(f"Time budget: {args.budget}s per method per P value")
    print(f"P values: {P_values}")

    results = {}
    for key in method_keys:
        name, func = ALL_EXPERIMENTS_R2[key]
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
    print("ROUND 2 SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Method':<40s} {'P=50':>10s} {'P=100':>10s} {'P=200':>10s}")
    print(f"{'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")

    # Add baseline for reference
    print(f"{'Baseline (LSE+Polyak)':<40s} {'1.519790':>10s} {'1.515085':>10s} {'1.512053':>10s}")

    for key in results:
        name = ALL_EXPERIMENTS_R2[key][0]
        vals = []
        for P in P_values:
            if P in results[key]:
                vals.append(f"{results[key][P]['value']:.6f}")
            else:
                vals.append("N/A")
        print(f"{name:<40s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_data_r2.json")
    save_results = {}
    for key in results:
        save_results[key] = {}
        for P in results[key]:
            r = results[key][P]
            save_results[key][str(P)] = {"value": r["value"], "restarts": r["restarts"], "elapsed": r["elapsed"]}
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
