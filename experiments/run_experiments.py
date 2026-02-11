"""
Optimization experiments for Sidon autocorrelation C_{1a}.

Fair comparison: every method gets the SAME wall-clock time budget per P value.
We fix P <= 200 for all experiments.
"""

import sys, os, time, json
import numpy as np

# Add parent dir so we can import sidon_core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidon_core import (
    autoconv_coeffs, project_simplex_nb, lse_obj_nb, lse_grad_nb,
    lse_objgrad_nb, armijo_step_nb_v2, _autoconv_max_argmax,
    _polyak_polish_nb, _hybrid_single_restart, make_inits,
    BETA_HEAVY, logsumexp_nb, softmax_nb
)

# ============================================================================
# Verification helper
# ============================================================================

def verify_solution(x, P):
    """Compute exact peak autoconvolution value for verification."""
    c = autoconv_coeffs(x, P)
    return float(np.max(c))


def save_solution(name, P, x, val, results_dir="experiments"):
    """Save best solution for later verification."""
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"solution_{name}_P{P}.json")
    data = {
        "method": name,
        "P": P,
        "value": val,
        "x": x.tolist()
    }
    with open(fname, "w") as f:
        json.dump(data, f)
    return fname


# ============================================================================
# Baseline: Hybrid LSE + Polyak (same as existing best method)
# ============================================================================

def run_baseline(P, time_budget):
    """Run hybrid LSE+Polyak for time_budget seconds."""
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    rng = np.random.default_rng(42)

    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x_init = rng.dirichlet(np.ones(P))
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
# Experiment 1: Mirror Descent (Entropic / KL proximal)
# ============================================================================

def run_mirror_descent(P, time_budget):
    """
    Mirror descent with KL divergence on the simplex.
    Uses multiplicative weights update instead of Euclidean projection.
    This naturally respects the simplex geometry.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Initialize
        x = rng.dirichlet(np.ones(P) * 2.0)
        n_restarts += 1

        # Beta continuation schedule
        for beta in BETA_HEAVY:
            eta = 0.5 / (beta + 1.0)  # step size decays with beta
            no_improve = 0
            stage_best = np.inf
            stage_best_x = x.copy()

            for t in range(3000):
                obj, g = lse_objgrad_nb(x, P, beta)

                # Mirror descent update: x_new[i] = x[i] * exp(-eta * g[i]) / Z
                log_x = np.log(np.maximum(x, 1e-30)) - eta * g
                log_x -= log_x.max()  # stability
                x_new = np.exp(log_x)
                x_new /= x_new.sum()

                x = x_new

                tv = float(np.max(autoconv_coeffs(x, P)))
                if tv < stage_best:
                    stage_best = tv
                    stage_best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > 500:
                    break

            x = stage_best_x

        # Polyak polish
        pol_val, pol_x = _polyak_polish_nb(x, P, 200000)

        if pol_val < best_val:
            best_val = pol_val
            best_x = pol_x.copy()

        if time.time() - t0 >= time_budget:
            break

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 2: Coordinate Descent (2-coordinate simplex moves)
# ============================================================================

def run_coordinate_descent(P, time_budget):
    """
    Coordinate descent: pick two coordinates, move mass between them
    to reduce the max autoconvolution. Very cheap per iteration.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Start from LSE-warmed point (quick LSE phase)
        x = rng.dirichlet(np.ones(P) * 2.0)
        n_restarts += 1
        beta_arr = np.array(BETA_HEAVY[:10], dtype=np.float64)  # short LSE warmup
        _, _, x = _hybrid_single_restart(x, P, beta_arr, 2000, 0)

        c = autoconv_coeffs(x, P)
        fval = float(np.max(c))
        local_best = fval
        local_best_x = x.copy()

        # Coordinate descent phase
        for t in range(500000):
            if time.time() - t0 >= time_budget:
                break

            c = autoconv_coeffs(x, P)
            k_star = int(np.argmax(c))
            fval = float(c[k_star])

            if fval < local_best:
                local_best = fval
                local_best_x = x.copy()

            # Pick coordinate i that contributes most to peak k_star
            # g[i] = 4P * x[k_star - i] for valid j = k_star - i
            contrib = np.zeros(P)
            for i in range(P):
                j = k_star - i
                if 0 <= j < P:
                    contrib[i] = x[j]

            # Move mass FROM the highest contributor TO a random other
            i_from = int(np.argmax(contrib))
            i_to = rng.integers(0, P)
            while i_to == i_from:
                i_to = rng.integers(0, P)

            # Try a small transfer
            delta = min(0.3 * x[i_from], 1.0 / P) if x[i_from] > 0 else 0
            if delta <= 0:
                continue

            x_trial = x.copy()
            x_trial[i_from] -= delta
            x_trial[i_to] += delta

            c_trial = autoconv_coeffs(x_trial, P)
            fval_trial = float(np.max(c_trial))

            if fval_trial < fval:
                x = x_trial

        if local_best < best_val:
            best_val = local_best
            best_x = local_best_x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 3: Differential Evolution
# ============================================================================

def run_differential_evolution(P, time_budget):
    """
    Differential Evolution on the simplex. Population-based search
    where solutions share information through crossover and mutation.
    """
    rng = np.random.default_rng(42)
    pop_size = 20
    F = 0.8  # mutation factor
    CR = 0.7  # crossover rate

    # Initialize population on simplex
    pop = np.array([rng.dirichlet(np.ones(P)) for _ in range(pop_size)])
    fitness = np.array([float(np.max(autoconv_coeffs(pop[i], P))) for i in range(pop_size)])

    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_x = pop[best_idx].copy()
    n_gens = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n_gens += 1
        for i in range(pop_size):
            if time.time() - t0 >= time_budget:
                break

            # DE/rand/1 mutation
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c_idx = rng.choice(idxs, 3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c_idx])

            # Crossover
            trial = pop[i].copy()
            j_rand = rng.integers(0, P)
            for j in range(P):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]

            # Project to simplex
            trial = np.maximum(trial, 0.0)
            if trial.sum() > 1e-12:
                trial /= trial.sum()
            else:
                trial = rng.dirichlet(np.ones(P))

            f_trial = float(np.max(autoconv_coeffs(trial, P)))

            if f_trial <= fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial

                if f_trial < best_val:
                    best_val = f_trial
                    best_x = trial.copy()

    elapsed = time.time() - t0

    # Polish best with Polyak
    pol_val, pol_x = _polyak_polish_nb(best_x, P, 50000)
    if pol_val < best_val:
        best_val = pol_val
        best_x = pol_x.copy()

    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_gens, elapsed


# ============================================================================
# Experiment 4: Frank-Wolfe (Conditional Gradient)
# ============================================================================

def run_frank_wolfe(P, time_budget):
    """
    Frank-Wolfe (conditional gradient) method. Linear minimization over
    the simplex is trivial (just pick the coordinate with smallest gradient).
    No projection needed! Natural for simplex constraints.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P) * 2.0)
        n_restarts += 1

        for beta in BETA_HEAVY:
            stage_best = np.inf
            stage_best_x = x.copy()
            no_improve = 0

            for t in range(5000):
                obj, g = lse_objgrad_nb(x, P, beta)

                # Frank-Wolfe direction: minimize <g, s> over simplex
                # => s = e_{argmin g}
                s = np.zeros(P)
                s[np.argmin(g)] = 1.0

                # Line search: minimize f(x + gamma*(s-x)) over gamma in [0,1]
                # Use simple step size gamma = 2/(t+2)
                gamma = 2.0 / (t + 3.0)
                x = x + gamma * (s - x)

                tv = float(np.max(autoconv_coeffs(x, P)))
                if tv < stage_best:
                    stage_best = tv
                    stage_best_x = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > 500:
                    break

            x = stage_best_x

        # Polyak polish
        pol_val, pol_x = _polyak_polish_nb(x, P, 200000)
        if pol_val < best_val:
            best_val = pol_val
            best_x = pol_x.copy()

        if time.time() - t0 >= time_budget:
            break

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 5: Simulated Annealing on Simplex
# ============================================================================

def run_simulated_annealing(P, time_budget):
    """
    Simulated annealing: accept worse solutions with probability depending
    on temperature. Uses simplex-preserving moves (swap mass between coords).
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n_restarts += 1
        x = rng.dirichlet(np.ones(P) * 2.0)
        fval = float(np.max(autoconv_coeffs(x, P)))
        local_best = fval
        local_best_x = x.copy()

        T_init = 0.05
        T_final = 1e-5
        n_steps = 200000
        for step in range(n_steps):
            if time.time() - t0 >= time_budget:
                break

            T = T_init * (T_final / T_init) ** (step / n_steps)

            # Propose: swap mass between two random coordinates
            i, j = rng.choice(P, 2, replace=False)
            max_delta = min(x[i], 0.5 / P)
            delta = rng.uniform(0, max_delta) if max_delta > 0 else 0
            if delta <= 0:
                continue

            x_new = x.copy()
            x_new[i] -= delta
            x_new[j] += delta

            fval_new = float(np.max(autoconv_coeffs(x_new, P)))

            # Accept/reject
            if fval_new < fval or rng.random() < np.exp(-(fval_new - fval) / T):
                x = x_new
                fval = fval_new
                if fval < local_best:
                    local_best = fval
                    local_best_x = x.copy()

        # Polish
        pol_val, pol_x = _polyak_polish_nb(local_best_x, P, 100000)
        if pol_val < best_val:
            best_val = pol_val
            best_x = pol_x.copy()

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 6: CMA-ES style (simplified covariance adaptation)
# ============================================================================

def run_cma_style(P, time_budget):
    """
    Simplified CMA-ES: maintain a mean and covariance on the simplex.
    Sample, evaluate, update mean toward better solutions.
    Uses log-space parameterization to stay positive, then normalize.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_gens = 0
    lam = 15  # population size
    mu = 5    # parents

    t0 = time.time()

    # Initialize mean in log-space
    mean = np.zeros(P)
    sigma = 0.5
    restarts = 0

    while time.time() - t0 < time_budget:
        restarts += 1
        mean = rng.standard_normal(P) * 0.3
        sigma = 0.5

        for gen in range(10000):
            if time.time() - t0 >= time_budget:
                break
            n_gens += 1

            # Sample population
            samples = []
            fitnesses = []
            for _ in range(lam):
                z = mean + sigma * rng.standard_normal(P)
                x = np.exp(z)
                x /= x.sum()
                f = float(np.max(autoconv_coeffs(x, P)))
                samples.append(z)
                fitnesses.append(f)

            # Sort by fitness
            order = np.argsort(fitnesses)

            # Check best
            x_best_gen = np.exp(samples[order[0]])
            x_best_gen /= x_best_gen.sum()
            if fitnesses[order[0]] < best_val:
                best_val = fitnesses[order[0]]
                best_x = x_best_gen.copy()

            # Update mean (weighted average of mu best)
            new_mean = np.zeros(P)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            for i in range(mu):
                new_mean += weights[i] * np.array(samples[order[i]])
            mean = new_mean

            # Adapt sigma (1/5 rule approximation)
            if fitnesses[order[0]] < fitnesses[order[lam // 2]]:
                sigma *= 1.02
            else:
                sigma *= 0.98
            sigma = max(sigma, 0.01)

    elapsed = time.time() - t0

    # Polish
    if best_x is not None:
        pol_val, pol_x = _polyak_polish_nb(best_x, P, 100000)
        if pol_val < best_val:
            best_val = pol_val
            best_x = pol_x.copy()

    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_gens, elapsed


# ============================================================================
# Experiment 7: Moreau Envelope Smoothing
# ============================================================================

def run_moreau_smoothing(P, time_budget):
    """
    Moreau envelope smoothing: instead of LSE, use the proximal smoothing
    f_mu(x) = min_y { max_k c_k(y) + ||x-y||^2 / (2*mu) }
    Approximate via gradient steps on the quadratic-penalized subproblem.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P) * 2.0)
        n_restarts += 1

        # Moreau continuation: decrease mu (tighter approximation)
        for mu in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]:
            # Inner loop: proximal gradient on max_k c_k(x) + regularization
            y = x.copy()
            stage_best = np.inf
            stage_best_x = x.copy()
            no_improve = 0

            for t in range(3000):
                c = autoconv_coeffs(y, P)
                k_star = int(np.argmax(c))
                fval = float(c[k_star])

                # Subgradient of max_k c_k at y
                g = np.zeros(P)
                for i in range(P):
                    j = k_star - i
                    if 0 <= j < P:
                        g[i] = 2.0 * (2.0 * P) * y[j]

                # Add proximal term gradient: (y - x) / mu
                g += (y - x) / mu

                # Step
                eta = 0.01 / (1.0 + t * 0.001)
                y = y - eta * g
                y = project_simplex_nb(y)

                tv = float(np.max(autoconv_coeffs(y, P)))
                if tv < stage_best:
                    stage_best = tv
                    stage_best_x = y.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > 300:
                    break

            x = stage_best_x

        # Polyak polish
        pol_val, pol_x = _polyak_polish_nb(x, P, 200000)
        if pol_val < best_val:
            best_val = pol_val
            best_x = pol_x.copy()

        if time.time() - t0 >= time_budget:
            break

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 8: Game-Theoretic (Min-Max with Adversarial k)
# ============================================================================

def run_game_theoretic(P, time_budget):
    """
    Treat max_k c_k as a two-player game: x-player minimizes, k-player maximizes.
    Run simultaneous gradient descent-ascent with softmax over k.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0
    K = 2 * P - 1  # number of autoconvolution coefficients

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P) * 2.0)
        # Initialize adversary weights (uniform over k)
        w = np.ones(K) / K
        n_restarts += 1

        local_best = np.inf
        local_best_x = x.copy()

        for t in range(100000):
            if time.time() - t0 >= time_budget:
                break

            c = autoconv_coeffs(x, P)
            fval = float(np.max(c))

            if fval < local_best:
                local_best = fval
                local_best_x = x.copy()

            # x-player: gradient descent on weighted objective sum_k w_k * c_k
            # This is like an adaptive weighted average of peaks
            g_x = np.zeros(P)
            for i in range(P):
                s = 0.0
                for j in range(P):
                    k = i + j
                    s += w[k] * x[j]
                g_x[i] = 2.0 * (2.0 * P) * s

            eta_x = 0.1 / (1.0 + t * 0.0001)
            x = x - eta_x * g_x
            x = project_simplex_nb(x)

            # k-player: multiplicative weights update to concentrate on high c_k
            eta_k = 0.1
            c = autoconv_coeffs(x, P)
            w = w * np.exp(eta_k * c)
            w /= w.sum()

        # Polish
        pol_val, pol_x = _polyak_polish_nb(local_best_x, P, 200000)
        if pol_val < best_val:
            best_val = pol_val
            best_x = pol_x.copy()

        if time.time() - t0 >= time_budget:
            break

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 9: Random Smoothing (Gaussian perturbation)
# ============================================================================

def run_random_smoothing(P, time_budget):
    """
    Randomized smoothing: estimate gradient of E[max_k c_k(x+noise)]
    via finite differences with Gaussian perturbations.
    The smoothed objective is differentiable even though max is not.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0
    n_samples = 10  # perturbation samples for gradient estimate

    t0 = time.time()
    while time.time() - t0 < time_budget:
        x = rng.dirichlet(np.ones(P) * 2.0)
        n_restarts += 1

        local_best = np.inf
        local_best_x = x.copy()

        # Continuation on sigma (smoothing radius)
        for sigma in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            for t in range(2000):
                if time.time() - t0 >= time_budget:
                    break

                # Estimate gradient via antithetic sampling
                g = np.zeros(P)
                for _ in range(n_samples):
                    z = rng.standard_normal(P)
                    x_plus = project_simplex_nb(x + sigma * z)
                    x_minus = project_simplex_nb(x - sigma * z)
                    f_plus = float(np.max(autoconv_coeffs(x_plus, P)))
                    f_minus = float(np.max(autoconv_coeffs(x_minus, P)))
                    g += (f_plus - f_minus) / (2.0 * sigma) * z

                g /= n_samples

                eta = 0.05 / (1.0 + t * 0.001)
                x = x - eta * g
                x = project_simplex_nb(x)

                fval = float(np.max(autoconv_coeffs(x, P)))
                if fval < local_best:
                    local_best = fval
                    local_best_x = x.copy()

        # Polish
        pol_val, pol_x = _polyak_polish_nb(local_best_x, P, 200000)
        if pol_val < best_val:
            best_val = pol_val
            best_x = pol_x.copy()

        if time.time() - t0 >= time_budget:
            break

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 10: Symmetric Exploitation
# ============================================================================

def run_symmetric_exploitation(P, time_budget):
    """
    Exploit the symmetry: autoconvolution c_k = c_{2(P-1)-k}.
    Restrict search to symmetric functions f(x) = f(-x), halving
    the effective dimension.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    half = P // 2

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Generate symmetric init
        x_half = rng.dirichlet(np.ones(half))
        x = np.zeros(P)
        x[:half] = x_half
        x[P - half:] = x_half[::-1]
        if P % 2 == 1:
            x[half] = rng.uniform(0.01, 0.1)
        x /= x.sum()
        n_restarts += 1

        # Run standard hybrid
        lse_v, pol_v, x_out = _hybrid_single_restart(
            x, P, beta_arr, n_iters_lse=15000, n_iters_polyak=200000)

        # Re-symmetrize and polish
        x_sym = (x_out + x_out[::-1]) / 2.0
        x_sym = project_simplex_nb(x_sym)
        pol_val, pol_x = _polyak_polish_nb(x_sym, P, 50000)

        # Also check unsymmetrized
        final_val = min(pol_v, pol_val)
        final_x = pol_x if pol_val <= pol_v else x_out

        if final_val < best_val:
            best_val = final_val
            best_x = final_x.copy()

        if time.time() - t0 >= time_budget:
            break

    elapsed = time.time() - t0
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 11: Multi-Peak Subgradient (cut all near-peaks simultaneously)
# ============================================================================

def run_multi_peak_subgradient(P, time_budget):
    """
    Instead of cutting just the argmax, identify ALL peaks within epsilon
    of the max, and take a subgradient step that pushes all of them down.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0
    beta_arr = np.array(BETA_HEAVY[:12], dtype=np.float64)  # short LSE warmup

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Quick LSE warmup
        x = rng.dirichlet(np.ones(P) * 2.0)
        n_restarts += 1
        _, _, x = _hybrid_single_restart(x, P, beta_arr, 3000, 0)

        local_best = np.inf
        local_best_x = x.copy()

        for t in range(500000):
            if time.time() - t0 >= time_budget:
                break

            c = autoconv_coeffs(x, P)
            fval = float(np.max(c))

            if fval < local_best:
                local_best = fval
                local_best_x = x.copy()

            # Find all peaks within eps of max
            eps = max(0.005 / (1.0 + t * 1e-5), 1e-4)
            peak_indices = np.where(c >= fval - eps)[0]

            # Average subgradient over all near-peak indices
            g = np.zeros(P)
            for k_star in peak_indices:
                for i in range(P):
                    j = k_star - i
                    if 0 <= j < P:
                        g[i] += 2.0 * (2.0 * P) * x[j]
            g /= len(peak_indices)

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
# Experiment 12: Particle Swarm Optimization
# ============================================================================

def run_pso(P, time_budget):
    """
    Particle Swarm Optimization on the simplex.
    Each particle has position, velocity, personal best, global best.
    """
    rng = np.random.default_rng(42)
    n_particles = 20
    w_inertia = 0.7
    c1, c2 = 1.5, 1.5  # cognitive and social

    # Init
    pos = np.array([rng.dirichlet(np.ones(P)) for _ in range(n_particles)])
    vel = rng.standard_normal((n_particles, P)) * 0.01
    fitness = np.array([float(np.max(autoconv_coeffs(pos[i], P))) for i in range(n_particles)])

    pbest = pos.copy()
    pbest_f = fitness.copy()
    gbest_idx = np.argmin(fitness)
    gbest = pos[gbest_idx].copy()
    gbest_f = fitness[gbest_idx]

    n_iters = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        n_iters += 1
        for i in range(n_particles):
            if time.time() - t0 >= time_budget:
                break

            r1, r2 = rng.random(P), rng.random(P)
            vel[i] = (w_inertia * vel[i]
                      + c1 * r1 * (pbest[i] - pos[i])
                      + c2 * r2 * (gbest - pos[i]))

            pos[i] = pos[i] + vel[i]
            pos[i] = np.maximum(pos[i], 0.0)
            s = pos[i].sum()
            if s > 1e-12:
                pos[i] /= s
            else:
                pos[i] = rng.dirichlet(np.ones(P))

            f = float(np.max(autoconv_coeffs(pos[i], P)))
            if f < pbest_f[i]:
                pbest_f[i] = f
                pbest[i] = pos[i].copy()
            if f < gbest_f:
                gbest_f = f
                gbest = pos[i].copy()

    elapsed = time.time() - t0

    # Polish
    pol_val, pol_x = _polyak_polish_nb(gbest, P, 100000)
    best_val = min(gbest_f, pol_val)
    best_x = pol_x if pol_val <= gbest_f else gbest

    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_iters, elapsed


# ============================================================================
# Experiment 13: Fourier Parameterization
# ============================================================================

def run_fourier_param(P, time_budget):
    """
    Parameterize f as f(x) = |sum_k a_k e^{2pi i k x}|^2, which is
    automatically nonnegative. Optimize over the Fourier coefficients.
    This changes the landscape entirely.
    """
    rng = np.random.default_rng(42)
    best_val = np.inf
    best_x = None
    n_restarts = 0
    n_freq = min(P // 4, 30)  # number of Fourier modes

    centers = np.linspace(-0.25, 0.25, P, endpoint=False) + 0.25 / P

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n_restarts += 1
        # Random Fourier coefficients (complex)
        a = rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)
        a[0] = abs(a[0])  # DC component real and positive

        local_best = np.inf
        local_best_x = None

        for beta in [1, 5, 20, 100, 500, 2000]:
            for t in range(2000):
                if time.time() - t0 >= time_budget:
                    break

                # Build x from Fourier coefficients
                # f(x) = |sum_k a_k exp(2pi i k x)|^2
                f_vals = np.zeros(P)
                for k in range(n_freq):
                    f_vals += np.real(a[k] * np.exp(2j * np.pi * k * centers * 2))  # freq scaled
                f_vals = f_vals ** 2  # square to ensure non-negativity
                f_vals = np.maximum(f_vals, 0.0)
                s = f_vals.sum()
                if s < 1e-12:
                    a = rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq)
                    continue
                x = f_vals / s

                fval = float(np.max(autoconv_coeffs(x, P)))
                if fval < local_best:
                    local_best = fval
                    local_best_x = x.copy()

                # Finite difference gradient on Fourier coefficients
                grad_a = np.zeros(n_freq, dtype=complex)
                eps = 1e-4
                for k in range(n_freq):
                    for part in [0, 1]:  # real, imag
                        a_plus = a.copy()
                        if part == 0:
                            a_plus[k] += eps
                        else:
                            a_plus[k] += eps * 1j

                        f_p = np.zeros(P)
                        for kk in range(n_freq):
                            f_p += np.real(a_plus[kk] * np.exp(2j * np.pi * kk * centers * 2))
                        f_p = f_p ** 2
                        f_p = np.maximum(f_p, 0.0)
                        sp = f_p.sum()
                        if sp < 1e-12:
                            continue
                        xp = f_p / sp
                        fval_p = lse_obj_nb(xp, P, beta)

                        if part == 0:
                            grad_a[k] += (fval_p - lse_obj_nb(x, P, beta)) / eps
                        else:
                            grad_a[k] += 1j * (fval_p - lse_obj_nb(x, P, beta)) / eps

                eta = 0.01 / (1.0 + t * 0.001)
                a = a - eta * grad_a

        # Polish best
        if local_best_x is not None:
            pol_val, pol_x = _polyak_polish_nb(local_best_x, P, 200000)
            if pol_val < best_val:
                best_val = pol_val
                best_x = pol_x.copy()

        if time.time() - t0 >= time_budget:
            break

    elapsed = time.time() - t0
    if best_x is None:
        best_x = np.ones(P) / P
    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_restarts, elapsed


# ============================================================================
# Experiment 14: Genetic Algorithm with Crossover
# ============================================================================

def run_genetic_algorithm(P, time_budget):
    """
    Genetic algorithm with simplex-preserving crossover.
    Tournament selection + arithmetic crossover + mutation.
    """
    rng = np.random.default_rng(42)
    pop_size = 30
    mutation_rate = 0.1
    tournament_size = 3

    # Initialize population
    pop = [rng.dirichlet(np.ones(P)) for _ in range(pop_size)]
    fitness = [float(np.max(autoconv_coeffs(x, P))) for x in pop]

    best_idx = np.argmin(fitness)
    best_val = fitness[best_idx]
    best_x = pop[best_idx].copy()
    n_gens = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n_gens += 1
        new_pop = []
        new_fitness = []

        # Elitism: keep top 2
        sorted_idx = np.argsort(fitness)
        for i in range(2):
            new_pop.append(pop[sorted_idx[i]].copy())
            new_fitness.append(fitness[sorted_idx[i]])

        while len(new_pop) < pop_size:
            if time.time() - t0 >= time_budget:
                break

            # Tournament selection
            def tournament():
                idxs = rng.choice(pop_size, tournament_size, replace=False)
                best_t = idxs[0]
                for idx in idxs[1:]:
                    if fitness[idx] < fitness[best_t]:
                        best_t = idx
                return pop[best_t]

            p1 = tournament()
            p2 = tournament()

            # Arithmetic crossover
            alpha = rng.uniform(0.2, 0.8)
            child = alpha * p1 + (1 - alpha) * p2

            # Mutation: add noise and re-normalize
            if rng.random() < mutation_rate:
                noise = rng.standard_normal(P) * 0.1 * np.mean(child)
                child = child + noise
                child = np.maximum(child, 0.0)
                s = child.sum()
                if s > 1e-12:
                    child /= s
                else:
                    child = rng.dirichlet(np.ones(P))

            f = float(np.max(autoconv_coeffs(child, P)))
            new_pop.append(child)
            new_fitness.append(f)

            if f < best_val:
                best_val = f
                best_x = child.copy()

        pop = new_pop[:pop_size]
        fitness = new_fitness[:pop_size]

    elapsed = time.time() - t0

    # Polish
    pol_val, pol_x = _polyak_polish_nb(best_x, P, 100000)
    if pol_val < best_val:
        best_val = pol_val
        best_x = pol_x.copy()

    exact_val = verify_solution(best_x, P)
    return exact_val, best_x, n_gens, elapsed


# ============================================================================
# Main experiment runner
# ============================================================================

ALL_EXPERIMENTS = {
    "baseline_lse_polyak": ("Hybrid LSE+Polyak (Baseline)", run_baseline),
    "mirror_descent": ("Mirror Descent (KL Proximal)", run_mirror_descent),
    "coordinate_descent": ("Coordinate Descent (2-swap)", run_coordinate_descent),
    "differential_evolution": ("Differential Evolution", run_differential_evolution),
    "frank_wolfe": ("Frank-Wolfe (Conditional Gradient)", run_frank_wolfe),
    "simulated_annealing": ("Simulated Annealing", run_simulated_annealing),
    "cma_style": ("CMA-ES (Log-space)", run_cma_style),
    "moreau_smoothing": ("Moreau Envelope Smoothing", run_moreau_smoothing),
    "game_theoretic": ("Game-Theoretic Min-Max", run_game_theoretic),
    "random_smoothing": ("Randomized Smoothing", run_random_smoothing),
    "symmetric_exploitation": ("Symmetric Exploitation", run_symmetric_exploitation),
    "multi_peak_subgradient": ("Multi-Peak Subgradient", run_multi_peak_subgradient),
    "pso": ("Particle Swarm Optimization", run_pso),
    "fourier_param": ("Fourier Parameterization", run_fourier_param),
    "genetic_algorithm": ("Genetic Algorithm", run_genetic_algorithm),
}


def run_all(P_values=(50, 100, 200), time_budget=60):
    """Run all experiments and print results."""
    print(f"=" * 70)
    print(f"Sidon Autocorrelation Optimization Experiments")
    print(f"Time budget: {time_budget}s per method per P value")
    print(f"P values: {P_values}")
    print(f"=" * 70)

    # Warmup Numba
    print("Warming up Numba JIT...")
    x_test = np.ones(5) / 5.0
    _ = autoconv_coeffs(x_test, 5)
    _ = _polyak_polish_nb(x_test, 5, 10)
    beta_arr = np.array([1.0, 10.0])
    _ = _hybrid_single_restart(x_test, 5, beta_arr, 10, 10)
    print("Warmup complete.\n")

    results = {}
    for method_key, (method_name, method_func) in ALL_EXPERIMENTS.items():
        results[method_key] = {}
        print(f"\n{'-' * 60}")
        print(f"Method: {method_name}")
        print(f"{'-' * 60}")

        for P in P_values:
            print(f"  P={P}...", end=" ", flush=True)
            val, x, restarts, elapsed = method_func(P, time_budget)
            results[method_key][P] = {
                "value": val,
                "restarts": restarts,
                "elapsed": elapsed,
                "x": x.tolist()
            }
            # Save solution
            save_solution(method_key, P, x, val)
            print(f"val={val:.6f}  restarts={restarts}  time={elapsed:.1f}s")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=60, help="Time budget per method per P")
    parser.add_argument("--P", type=str, default="50,100,200", help="P values (comma-separated)")
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated method keys or 'all'")
    args = parser.parse_args()

    P_values = tuple(int(p) for p in args.P.split(","))

    if args.methods == "all":
        results = run_all(P_values, args.budget)
    else:
        method_keys = args.methods.split(",")
        print(f"Running methods: {method_keys}")
        print(f"Time budget: {args.budget}s per method per P value")
        print(f"P values: {P_values}")

        # Warmup
        print("Warming up Numba JIT...")
        x_test = np.ones(5) / 5.0
        _ = autoconv_coeffs(x_test, 5)
        _ = _polyak_polish_nb(x_test, 5, 10)
        beta_arr = np.array([1.0, 10.0])
        _ = _hybrid_single_restart(x_test, 5, beta_arr, 10, 10)
        print("Warmup complete.\n")

        results = {}
        for key in method_keys:
            name, func = ALL_EXPERIMENTS[key]
            results[key] = {}
            print(f"\n{'-' * 60}")
            print(f"Method: {name}")
            print(f"{'-' * 60}")
            for P in P_values:
                print(f"  P={P}...", end=" ", flush=True)
                val, x, restarts, elapsed = func(P, args.budget)
                results[key][P] = {"value": val, "restarts": restarts, "elapsed": elapsed, "x": x.tolist()}
                save_solution(key, P, x, val)
                print(f"val={val:.6f}  restarts={restarts}  time={elapsed:.1f}s")

    # Print summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Method':<35s} {'P=50':>10s} {'P=100':>10s} {'P=200':>10s}")
    print(f"{'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10}")
    for key in results:
        name = ALL_EXPERIMENTS[key][0]
        vals = []
        for P in P_values:
            if P in results[key]:
                vals.append(f"{results[key][P]['value']:.6f}")
            else:
                vals.append("N/A")
        print(f"{name:<35s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")

    # Save results JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_data.json")
    # Convert for JSON (strip numpy arrays)
    save_results = {}
    for key in results:
        save_results[key] = {}
        for P in results[key]:
            r = results[key][P]
            save_results[key][str(P)] = {
                "value": r["value"],
                "restarts": r["restarts"],
                "elapsed": r["elapsed"]
            }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
