"""
Session 2 experiments: focused on beating P=200 baseline (1.510357).

Strategy: the baseline's best P=200 result came from pure random restarts with
enough compute. We need methods that find BETTER basins, not just more restarts.

New ideas NOT tried in session 1:
1. Solution surgery: analyze best solutions, identify active bins, focus search there
2. Multi-resolution polish: P=100 -> P=200 with careful upsampling + heavy polish
3. Ensemble averaging of top-K solutions then polish
4. Block coordinate moves (adjacent bins together)
5. Spectral init from Fourier analysis of good solutions
6. Perturbation radius sweep (systematic, not random)
7. Solution blending from different methods
8. Very long single-restart optimization (all budget on one run)
9. Iterated warm restart (polish -> perturb -> polish -> perturb -> ...)
10. LP-inspired direction: move toward a solution with fewer active peaks
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidon_core import (
    autoconv_coeffs, project_simplex_nb, lse_obj_nb, lse_grad_nb,
    lse_objgrad_nb, armijo_step_nb_v2, _autoconv_max_argmax,
    _polyak_polish_nb, _hybrid_single_restart, _cyclic_polish_nb,
    make_inits, upsample_solution, BETA_HEAVY, BETA_ULTRA,
    logsumexp_nb, softmax_nb
)

P = 200  # Fixed for all experiments


def verify_solution(x):
    c = autoconv_coeffs(x, P)
    return float(np.max(c))


def save_solution(name, x, val):
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"solution_s2_{name}_P{P}.json")
    data = {"method": name, "P": P, "value": val, "x": x.tolist()}
    with open(fname, "w") as f:
        json.dump(data, f)


def load_best_solution():
    """Load the best known P=200 solution."""
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "solution_final_final_baseline_P200.json")
    with open(fname) as f:
        data = json.load(f)
    return np.array(data['x'])


def load_all_good_solutions(threshold=1.515):
    """Load all P=200 solutions below threshold."""
    import glob
    sols = []
    for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "solution_*_P200.json")):
        with open(f) as fh:
            data = json.load(fh)
        x = np.array(data['x'])
        if x.sum() < 0.5:
            continue
        v = verify_solution(x)
        if v < threshold:
            sols.append((v, x, f))
    sols.sort(key=lambda t: t[0])
    return sols


# ============================================================================
# Baseline for comparison at same budget
# ============================================================================

def run_baseline(time_budget, seed=42):
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
    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-1: Iterated Warm Restart (polish -> perturb -> polish -> ...)
# ============================================================================

def run_iterated_warm(time_budget, seed=42):
    """
    Start from best known solution. Repeatedly:
    1. Perturb with calibrated noise
    2. Full LSE+Polyak optimization
    3. If better, update warm start

    Key difference from R3 warm_from_best: also tries MANY perturbation scales
    systematically and uses the ULTRA beta schedule for deeper optimization.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)
    warm_x = load_best_solution()
    best_val = verify_solution(warm_x)
    best_x = warm_x.copy()
    n = 0

    # Perturbation scale schedule: try different scales systematically
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
            warm_x = x.copy()  # Update warm start

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-2: Ensemble Averaging + Polish
# ============================================================================

def run_ensemble_average(time_budget, seed=42):
    """
    Load all good P=200 solutions, compute weighted average (lower val = higher weight),
    then use as starting point for heavy optimization.
    Also try blending pairs of top solutions.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)
    sols = load_all_good_solutions(1.516)

    if len(sols) < 2:
        return run_baseline(time_budget, seed)

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()

    # Try various blends
    while time.time() - t0 < time_budget:
        n += 1
        if n <= 3 and len(sols) >= 3:
            # Weighted average of top-K
            k = min(n + 2, len(sols))
            vals = np.array([s[0] for s in sols[:k]])
            weights = vals.max() - vals + 1e-6
            weights /= weights.sum()
            x_init = sum(w * sols[i][1] for i, w in enumerate(weights[:k]))
            x_init += 0.1 * rng.standard_normal(P) * np.mean(x_init)
        elif len(sols) >= 2:
            # Random pair blend
            i, j = rng.choice(min(8, len(sols)), 2, replace=False)
            alpha = rng.beta(2, 2)
            x_init = alpha * sols[i][1] + (1 - alpha) * sols[j][1]
            noise = rng.uniform(0.05, 0.5)
            x_init += noise * rng.standard_normal(P) * np.mean(x_init)
        else:
            x_init = rng.dirichlet(np.ones(P))

        x_init = np.maximum(x_init, 0.0)
        if x_init.sum() < 1e-12:
            x_init = rng.dirichlet(np.ones(P))
        else:
            x_init /= x_init.sum()

        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-3: Multi-Resolution Polish (P=50 -> P=100 -> P=200)
# ============================================================================

def run_multiresolution(time_budget, seed=42):
    """
    3-level resolution cascade with heavy exploration at each level:
    P=50 (20% budget) -> P=100 (30% budget) -> P=200 (50% budget)
    At each level, keep top-3 solutions and upsample all.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()

    # Phase 1: P=50 exploration (20% budget)
    pool_50 = []
    while time.time() - t0 < time_budget * 0.2:
        choice = n % 4
        if choice == 0:
            x_init = rng.dirichlet(np.full(50, 0.05))
        elif choice == 1:
            z = np.abs(rng.standard_cauchy(50))
            x_init = z / z.sum()
        elif choice == 2:
            x_init = rng.random(50) ** rng.uniform(3, 10)
            x_init /= x_init.sum()
        else:
            x_init = rng.dirichlet(np.ones(50))
        _, pol_v, x = _hybrid_single_restart(x_init, 50, beta_arr, 15000, 200000)
        n += 1
        pool_50.append((pol_v, x.copy()))

    pool_50.sort(key=lambda t: t[0])
    top3_50 = pool_50[:3]

    # Phase 2: P=100 from upsampled P=50 + fresh (30% budget)
    pool_100 = []
    while time.time() - t0 < time_budget * 0.5:
        if rng.random() < 0.7 and len(top3_50) > 0:
            idx = rng.integers(0, len(top3_50))
            x_init = upsample_solution(top3_50[idx][1], 50, 100)
            x_init += rng.uniform(0.1, 0.4) * rng.standard_normal(100) * np.mean(x_init)
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        else:
            choice = rng.integers(0, 4)
            if choice == 0:
                x_init = rng.dirichlet(np.full(100, 0.05))
            elif choice == 1:
                z = np.abs(rng.standard_cauchy(100))
                x_init = z / z.sum()
            else:
                x_init = rng.dirichlet(np.ones(100))
        _, pol_v, x = _hybrid_single_restart(x_init, 100, beta_arr, 15000, 200000)
        n += 1
        pool_100.append((pol_v, x.copy()))

    pool_100.sort(key=lambda t: t[0])
    top3_100 = pool_100[:3]

    # Phase 3: P=200 from upsampled P=100 + fresh heavy-tail (50% budget)
    while time.time() - t0 < time_budget:
        if rng.random() < 0.7 and len(top3_100) > 0:
            idx = rng.integers(0, len(top3_100))
            x_init = upsample_solution(top3_100[idx][1], 100, P)
            x_init += rng.uniform(0.05, 0.3) * rng.standard_normal(P) * np.mean(x_init)
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        else:
            choice = rng.integers(0, 4)
            if choice == 0:
                x_init = rng.dirichlet(np.full(P, 0.05))
            elif choice == 1:
                z = np.abs(rng.standard_cauchy(P))
                x_init = z / z.sum()
            else:
                x_init = rng.dirichlet(np.ones(P))
        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-4: Solution Surgery (analyze + refine active bins)
# ============================================================================

def run_solution_surgery(time_budget, seed=42):
    """
    Analyze the best solution's structure:
    - Identify active bins (x_i > threshold)
    - Create new inits that have the same sparsity pattern but different values
    - Run LSE+Polyak from these structured inits
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)
    warm_x = load_best_solution()

    # Analyze structure
    threshold = np.mean(warm_x) * 0.1
    active = warm_x > threshold
    n_active = np.sum(active)

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n += 1
        strategy = n % 5

        if strategy == 0:
            # Same sparsity pattern, perturbed values
            x_init = np.zeros(P)
            x_init[active] = warm_x[active] * (1 + 0.3 * rng.standard_normal(n_active))
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        elif strategy == 1:
            # Same pattern, Dirichlet values on active bins
            x_init = np.zeros(P)
            x_init[active] = rng.dirichlet(np.ones(n_active) * 2)
            x_init /= x_init.sum()
        elif strategy == 2:
            # Slightly expanded pattern (add neighbors of active bins)
            expanded = active.copy()
            for i in range(P):
                if active[i]:
                    if i > 0: expanded[i-1] = True
                    if i < P-1: expanded[i+1] = True
            n_exp = np.sum(expanded)
            x_init = np.zeros(P)
            x_init[expanded] = rng.dirichlet(np.ones(n_exp))
            x_init /= x_init.sum()
        elif strategy == 3:
            # Shifted pattern (shift active bins by 1-3 positions)
            shift = rng.integers(-3, 4)
            x_init = np.zeros(P)
            for i in range(P):
                j = i + shift
                if 0 <= j < P and active[i]:
                    x_init[j] = warm_x[i]
            x_init = np.maximum(x_init, 0.0)
            if x_init.sum() > 1e-12:
                x_init /= x_init.sum()
            else:
                x_init = rng.dirichlet(np.ones(P))
        else:
            # Heavy-tail init (baseline comparison within this experiment)
            z = np.abs(rng.standard_cauchy(P))
            x_init = z / z.sum()

        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-5: Perturbation Scale Sweep (systematic)
# ============================================================================

def run_perturbation_sweep(time_budget, seed=42):
    """
    Systematic sweep of perturbation scales from the best known solution.
    For each scale, do multiple restarts. Track which scale works best.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    warm_x = load_best_solution()

    best_val = np.inf
    best_x = None
    n = 0

    # 16 perturbation scales from very small to very large
    scales = np.logspace(-2, 0.5, 16)  # 0.01 to ~3.16

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

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-6: Very Long Single Run (all budget on one optimization)
# ============================================================================

def run_very_long_single(time_budget, seed=42):
    """
    Instead of many restarts, do ONE very deep optimization:
    - Extended beta schedule (more stages, higher max)
    - 100K LSE iters per stage (vs 15K default)
    - 2M Polyak iters (vs 200K default)
    This tests whether depth beats breadth.
    """
    rng = np.random.default_rng(seed)
    # Extended beta schedule
    beta_extended = np.array([
        0.5, 1, 1.5, 2, 3, 4, 6, 9, 13, 20, 30, 45, 70, 100,
        150, 220, 330, 500, 750, 1100, 1600, 2400, 3500, 5000, 8000
    ], dtype=np.float64)

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Use heavy-tailed init for the one shot
        choice = n % 3
        if choice == 0:
            x_init = rng.dirichlet(np.full(P, 0.05))
        elif choice == 1:
            z = np.abs(rng.standard_cauchy(P))
            x_init = z / z.sum()
        else:
            # Start from best known + small perturbation
            warm_x = load_best_solution()
            x_init = warm_x + 0.2 * rng.standard_normal(P) * np.mean(warm_x)
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()

        _, pol_v, x = _hybrid_single_restart(
            x_init, P, beta_extended,
            n_iters_lse=50000, n_iters_polyak=1000000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-7: Spectral Initialization
# ============================================================================

def run_spectral_init(time_budget, seed=42):
    """
    Analyze the Fourier spectrum of the best solution, then create inits
    that match the dominant frequencies but with random phases.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    warm_x = load_best_solution()

    # Compute FFT of best solution
    fft_warm = np.fft.rfft(warm_x)
    magnitudes = np.abs(fft_warm)

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n += 1
        if n % 3 != 2:
            # Spectral init: same magnitudes, random phases
            phases = rng.uniform(0, 2 * np.pi, len(fft_warm))
            phases[0] = 0  # DC component stays real
            # Perturb magnitudes slightly
            mag_perturbed = magnitudes * (1 + 0.3 * rng.standard_normal(len(magnitudes)))
            mag_perturbed = np.maximum(mag_perturbed, 0)

            fft_new = mag_perturbed * np.exp(1j * phases)
            x_init = np.fft.irfft(fft_new, n=P)
            x_init = np.maximum(x_init, 0.0)
            if x_init.sum() < 1e-12:
                x_init = rng.dirichlet(np.ones(P))
            else:
                x_init /= x_init.sum()
        else:
            # Heavy-tail for diversity
            z = np.abs(rng.standard_cauchy(P))
            x_init = z / z.sum()

        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-8: Block Coordinate Descent + LSE
# ============================================================================

def run_block_coordinate(time_budget, seed=42):
    """
    After LSE warmup, do block coordinate moves: pick a block of 5-20
    adjacent bins and optimize them while holding the rest fixed.
    Uses projected gradient within the block.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Start from heavy-tailed init
        choice = n % 3
        if choice == 0:
            x_init = rng.dirichlet(np.full(P, 0.05))
        elif choice == 1:
            z = np.abs(rng.standard_cauchy(P))
            x_init = z / z.sum()
        else:
            x_init = rng.dirichlet(np.ones(P))
        n += 1

        # LSE warmup
        _, _, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 0)

        # Block coordinate refinement
        local_best = float(np.max(autoconv_coeffs(x, P)))
        local_best_x = x.copy()

        block_size = 15
        for sweep in range(50):
            if time.time() - t0 >= time_budget:
                break

            # Random block start
            start = rng.integers(0, P - block_size)
            block = slice(start, start + block_size)

            for inner in range(200):
                c = autoconv_coeffs(x, P)
                fval = float(np.max(c))
                k_star = int(np.argmax(c))

                if fval < local_best:
                    local_best = fval
                    local_best_x = x.copy()

                # Gradient only for block coordinates
                g_block = np.zeros(block_size)
                scale4P = 2.0 * (2.0 * P)
                for bi, i in enumerate(range(start, start + block_size)):
                    j = k_star - i
                    if 0 <= j < P:
                        g_block[bi] = scale4P * x[j]

                gnorm2 = float(np.dot(g_block, g_block))
                if gnorm2 < 1e-20:
                    break

                offset = 0.005 / (1.0 + sweep * 0.01)
                target = local_best - offset
                step = (fval - target) / gnorm2
                if step < 0:
                    step = 1e-5

                x[block] -= step * g_block
                x = project_simplex_nb(x)

        # Final Polyak polish
        pol_val, pol_x = _polyak_polish_nb(local_best_x, P, 100000)
        final_val = min(local_best, pol_val)
        final_x = pol_x if pol_val <= local_best else local_best_x

        if final_val < best_val:
            best_val = final_val
            best_x = final_x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-9: LP Rounding (relaxation-guided initialization)
# ============================================================================

def run_lp_rounding(time_budget, seed=42):
    """
    Use LP relaxation idea: for each k, c_k = 2P * sum_{i+j=k} x_i*x_j <= t.
    The constraint c_k <= t for all k with x on simplex is NOT an LP (quadratic).
    But we can linearize around a good solution:
    c_k(x+d) ≈ c_k(x) + grad_k . d
    and solve the LP to find a descent direction d.

    Approximate: find direction that reduces the max peak most.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Heavy-tailed init
        choice = n % 3
        if choice == 0:
            x_init = rng.dirichlet(np.full(P, 0.05))
        elif choice == 1:
            z = np.abs(rng.standard_cauchy(P))
            x_init = z / z.sum()
        else:
            x_init = rng.dirichlet(np.ones(P))
        n += 1

        # Full LSE warmup
        _, _, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 0)

        # LP-guided refinement: iteratively find the vertex of the simplex
        # that most reduces the weighted peak, then move toward it
        local_best = float(np.max(autoconv_coeffs(x, P)))
        local_best_x = x.copy()

        for t in range(50000):
            if time.time() - t0 >= time_budget:
                break

            c = autoconv_coeffs(x, P)
            fval = float(np.max(c))

            if fval < local_best:
                local_best = fval
                local_best_x = x.copy()

            # Compute gradient of weighted top peaks
            top_k = 3
            top_idx = np.argpartition(c, -top_k)[-top_k:]

            g = np.zeros(P)
            scale4P = 2.0 * (2.0 * P)
            for ki in top_idx:
                w = c[ki] / c[top_idx].sum()  # weight by peak value
                for i in range(P):
                    j = ki - i
                    if 0 <= j < P:
                        g[i] += w * scale4P * x[j]

            # LP direction: minimize g^T d s.t. d on simplex
            # => move toward e_{argmin g}
            i_min = np.argmin(g)

            # Frank-Wolfe step
            gamma = 2.0 / (t + 3.0)
            x_new = (1 - gamma) * x
            x_new[i_min] += gamma

            c_new = autoconv_coeffs(x_new, P)
            if float(np.max(c_new)) < fval:
                x = x_new
            else:
                # Subgradient fallback
                gnorm2 = float(np.dot(g, g))
                if gnorm2 < 1e-20:
                    break
                offset = 0.005 / (1.0 + t * 1e-4)
                target = local_best - offset
                step = (fval - target) / gnorm2
                if step < 0:
                    step = 1e-5 / (1.0 + t * 1e-4)
                x = x - step * g
                x = project_simplex_nb(x)

        # Polyak polish
        pol_val, pol_x = _polyak_polish_nb(local_best_x, P, 100000)
        final_val = min(local_best, pol_val)
        final_x = pol_x if pol_val <= local_best else local_best_x

        if final_val < best_val:
            best_val = final_val
            best_x = final_x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# S2-10: Heavy-Tail Elite Breeding (refined from R3)
# ============================================================================

def run_heavy_elite_v2(time_budget, seed=42):
    """
    Refined elite breeding: larger pool (12), more aggressive breeding rate (80%),
    mix of Cauchy mutations and crossover. Use ULTRA beta schedule.
    """
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
            # Breed from pool
            vals = np.array([p[0] for p in pool])
            probs = vals.max() - vals + 1e-8
            probs /= probs.sum()

            breed_type = rng.integers(0, 4)
            if breed_type == 0:
                # Pair crossover with Cauchy mutation
                idx1, idx2 = rng.choice(len(pool), 2, replace=False, p=probs)
                alpha = rng.beta(2, 2)
                x_init = alpha * pool[idx1][1] + (1 - alpha) * pool[idx2][1]
                x_init += rng.standard_cauchy(P) * 0.015
            elif breed_type == 1:
                # Triple blend
                idx = rng.choice(len(pool), 3, replace=False, p=probs)
                w = rng.dirichlet(np.ones(3))
                x_init = sum(w[i] * pool[idx[i]][1] for i in range(3))
                x_init += 0.2 * rng.standard_normal(P) * np.mean(x_init)
            elif breed_type == 2:
                # Best solution + sparse perturbation
                best_pool_idx = np.argmin(vals)
                x_init = pool[best_pool_idx][1].copy()
                k = max(3, P // 8)
                idx_p = rng.choice(P, k, replace=False)
                x_init[idx_p] *= np.exp(0.5 * rng.standard_normal(k))
            else:
                # Random pool member + heavy noise
                idx = rng.choice(len(pool), p=probs)
                x_init = pool[idx][1] + 0.5 * rng.standard_normal(P) * np.mean(pool[idx][1])

            x_init = np.maximum(x_init, 0.0)
            if x_init.sum() < 1e-12:
                x_init = rng.dirichlet(np.ones(P))
            else:
                x_init /= x_init.sum()
        else:
            # Fresh heavy-tailed init
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


# ============================================================================
# S2-11: Mirrored Sampling (antithetic warm restarts)
# ============================================================================

def run_mirrored_sampling(time_budget, seed=42):
    """
    For each perturbation z, try both x+z and x-z (antithetic pair).
    This doubles the effective exploration per unit randomness and
    reduces variance of the search.
    """
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

        # Try x + z
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

        # Try x - z
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


# ============================================================================
# S2-12: Adaptive Perturbation (learn which perturbation scale works)
# ============================================================================

def run_adaptive_perturbation(time_budget, seed=42):
    """
    Track which perturbation scales lead to improvements and adapt.
    Use a bandit-like approach: scales that produced improvements get
    sampled more often.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)
    warm_x = load_best_solution()

    best_val = verify_solution(warm_x)
    best_x = warm_x.copy()
    n = 0

    # Bandit over perturbation scales
    n_arms = 8
    scales = np.logspace(-2, 0.3, n_arms)  # 0.01 to ~2
    successes = np.ones(n_arms)  # prior
    failures = np.ones(n_arms)

    t0 = time.time()
    while time.time() - t0 < time_budget:
        # Thompson sampling: pick scale
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


# ============================================================================
# Main runner
# ============================================================================

ALL_EXPERIMENTS = {
    "baseline": ("Baseline LSE+Polyak", run_baseline),
    "iterated_warm": ("Iterated Warm Restart", run_iterated_warm),
    "ensemble_avg": ("Ensemble Averaging", run_ensemble_average),
    "multiresolution": ("Multi-Resolution Cascade", run_multiresolution),
    "solution_surgery": ("Solution Surgery", run_solution_surgery),
    "perturbation_sweep": ("Perturbation Scale Sweep", run_perturbation_sweep),
    "very_long_single": ("Very Long Single Run", run_very_long_single),
    "spectral_init": ("Spectral Initialization", run_spectral_init),
    "block_coordinate": ("Block Coordinate Descent", run_block_coordinate),
    "lp_rounding": ("LP-Guided Refinement", run_lp_rounding),
    "heavy_elite_v2": ("Heavy-Tail Elite v2", run_heavy_elite_v2),
    "mirrored_sampling": ("Mirrored Sampling", run_mirrored_sampling),
    "adaptive_perturbation": ("Adaptive Perturbation", run_adaptive_perturbation),
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--n_trials", type=int, default=1)
    args = parser.parse_args()

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
        method_keys = list(ALL_EXPERIMENTS.keys())
    else:
        method_keys = args.methods.split(",")

    print(f"Running P={P}, methods: {method_keys}")
    print(f"Budget: {args.budget}s, Trials: {args.n_trials}")
    print()

    results = {}
    for key in method_keys:
        name, func = ALL_EXPERIMENTS[key]
        print(f"\n{'-' * 60}")
        print(f"Method: {name}")
        print(f"{'-' * 60}")

        trials = []
        for trial in range(args.n_trials):
            seed = 42 + trial * 1000
            if args.n_trials > 1:
                print(f"  Trial {trial+1}/{args.n_trials}...", end=" ", flush=True)
            else:
                print(f"  Running...", end=" ", flush=True)

            val, x, restarts, elapsed = func(args.budget, seed=seed)
            trials.append({"value": val, "restarts": restarts, "elapsed": elapsed})
            save_solution(f"{key}_t{trial}", x, val)
            print(f"val={val:.6f}  restarts={restarts}  time={elapsed:.1f}s")

        vals = [t["value"] for t in trials]
        results[key] = {
            "best": min(vals),
            "mean": np.mean(vals),
            "std": np.std(vals) if len(vals) > 1 else 0,
            "trials": trials
        }

    # Summary
    print(f"\n\n{'=' * 70}")
    print(f"SESSION 2 SUMMARY (P={P})")
    print(f"{'=' * 70}")
    print(f"{'Method':<35s} {'Best':>10s} {'vs Baseline':>12s} {'Restarts':>10s}")
    print(f"{'-' * 35} {'-' * 10} {'-' * 12} {'-' * 10}")

    baseline_val = results.get("baseline", {}).get("best", 1.512053)
    for key in results:
        name = ALL_EXPERIMENTS[key][0]
        best = results[key]["best"]
        delta = best - baseline_val
        restarts = results[key]["trials"][0]["restarts"]
        marker = " ***" if delta < -0.0001 else ""
        print(f"{name:<35s} {best:10.6f} {delta:+12.6f} {restarts:>10d}{marker}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_data_s2.json")
    save_results = {}
    for key in results:
        save_results[key] = results[key]
        # Remove x arrays from trials for smaller JSON
        for t in save_results[key]["trials"]:
            pass  # already no x in trials dict
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to {out_path}")
