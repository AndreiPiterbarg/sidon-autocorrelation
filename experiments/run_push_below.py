"""
Push below 1.5104: combine the best session 2 findings.

Key insight: warm restart methods all converge to the same 1.51036 basin.
To push further, we need to either:
1. Find a DIFFERENT basin (better init diversity)
2. Polish the known basin more aggressively (longer Polyak, averaging)
3. Use the warm solution structure but break out of its basin
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidon_core import (
    autoconv_coeffs, project_simplex_nb,
    _polyak_polish_nb, _hybrid_single_restart, _cyclic_polish_nb,
    upsample_solution, BETA_HEAVY, BETA_ULTRA
)

P = 200


def verify_solution(x):
    c = autoconv_coeffs(x, P)
    return float(np.max(c))


def save_solution(name, x, val):
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"solution_push_{name}_P{P}.json")
    data = {"method": name, "P": P, "value": val, "x": x.tolist()}
    with open(fname, "w") as f:
        json.dump(data, f)


def load_best_solution():
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "solution_final_final_baseline_P200.json")
    with open(fname) as f:
        data = json.load(f)
    return np.array(data['x'])


# ============================================================================
# P1: Super Polish (very long Polyak + cyclic on best known)
# ============================================================================

def run_super_polish(time_budget, seed=42):
    """
    Take the best known solution and polish it with 2M Polyak iterations,
    then 2M cyclic iterations. Pure refinement, no restart.
    """
    warm_x = load_best_solution()
    best_val = verify_solution(warm_x)
    best_x = warm_x.copy()

    t0 = time.time()
    # Very long Polyak
    pol_val, pol_x = _polyak_polish_nb(warm_x, P, 2000000)
    if pol_val < best_val:
        best_val = pol_val
        best_x = pol_x.copy()
        print(f"  After 2M Polyak: {best_val:.8f}")

    # Very long cyclic
    if time.time() - t0 < time_budget:
        cyc_val, cyc_x = _cyclic_polish_nb(best_x, P, 2000000)
        if cyc_val < best_val:
            best_val = cyc_val
            best_x = cyc_x.copy()
            print(f"  After 2M cyclic: {best_val:.8f}")

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, 0, elapsed


# ============================================================================
# P2: Extreme Heavy-Tail Exploration (very sparse inits)
# ============================================================================

def run_extreme_sparse(time_budget, seed=42):
    """
    Use EXTREMELY sparse initializations: only 3-10 active bins out of 200.
    This explores a completely different part of the simplex that normal
    methods never reach.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n += 1
        k = rng.integers(3, 15)  # very few active bins
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
# P3: Multiple Warm Solutions + Heavy Polish
# ============================================================================

def run_multi_warm_heavy_polish(time_budget, seed=42):
    """
    Load ALL good P=200 solutions, perturb each, run with ULTRA beta
    and extra-long Polyak (500K). Focus on deep polish over breadth.
    """
    import glob
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)

    # Load top solutions
    sols = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "solution_*_P200.json"))):
        with open(f) as fh:
            data = json.load(fh)
        x = np.array(data['x'])
        if x.sum() < 0.5:
            continue
        v = verify_solution(x)
        if v < 1.513:
            sols.append((v, x))
    sols.sort(key=lambda t: t[0])

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()
    while time.time() - t0 < time_budget:
        n += 1
        if len(sols) > 0:
            # Pick a good solution and perturb
            idx = rng.integers(0, min(5, len(sols)))
            scale = rng.uniform(0.05, 0.6)
            x_init = sols[idx][1] + scale * rng.standard_normal(P) * np.mean(sols[idx][1])
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        else:
            x_init = rng.dirichlet(np.full(P, 0.05))

        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 20000, 500000)
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# P4: Low-P Basin Hunting + Careful Upsampling
# ============================================================================

def run_basin_hunt_upsample(time_budget, seed=42):
    """
    Massive exploration at P=30 and P=50 with very many restarts,
    then carefully upsample the SINGLE best to P=200 and do heavy polish.
    """
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_HEAVY, dtype=np.float64)

    best_val = np.inf
    best_x = None
    n = 0

    t0 = time.time()

    # Phase 1: massive P=30 exploration (15% budget)
    pool_30 = []
    while time.time() - t0 < time_budget * 0.15:
        choice = n % 4
        if choice == 0:
            x_init = rng.dirichlet(np.full(30, 0.05))
        elif choice == 1:
            z = np.abs(rng.standard_cauchy(30))
            x_init = z / z.sum()
        elif choice == 2:
            x_init = rng.random(30) ** rng.uniform(5, 15)
            x_init /= x_init.sum()
        else:
            x_init = rng.dirichlet(np.ones(30))
        _, pol_v, x = _hybrid_single_restart(x_init, 30, beta_arr, 15000, 200000)
        n += 1
        pool_30.append((pol_v, x.copy()))

    pool_30.sort(key=lambda t: t[0])

    # Phase 2: massive P=50 from top-5 P=30 + fresh (15% budget)
    pool_50 = []
    top5_30 = pool_30[:5]
    while time.time() - t0 < time_budget * 0.30:
        if rng.random() < 0.6 and len(top5_30) > 0:
            idx = rng.integers(0, len(top5_30))
            x_init = upsample_solution(top5_30[idx][1], 30, 50)
            x_init += rng.uniform(0.1, 0.3) * rng.standard_normal(50) * np.mean(x_init)
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        else:
            choice = rng.integers(0, 3)
            if choice == 0:
                x_init = rng.dirichlet(np.full(50, 0.05))
            elif choice == 1:
                z = np.abs(rng.standard_cauchy(50))
                x_init = z / z.sum()
            else:
                x_init = rng.dirichlet(np.ones(50))
        _, pol_v, x = _hybrid_single_restart(x_init, 50, beta_arr, 15000, 200000)
        n += 1
        pool_50.append((pol_v, x.copy()))

    pool_50.sort(key=lambda t: t[0])

    # Phase 3: P=100 from top-3 P=50 (20% budget)
    pool_100 = []
    top3_50 = pool_50[:3]
    while time.time() - t0 < time_budget * 0.50:
        if rng.random() < 0.7 and len(top3_50) > 0:
            idx = rng.integers(0, len(top3_50))
            x_init = upsample_solution(top3_50[idx][1], 50, 100)
            x_init += rng.uniform(0.05, 0.2) * rng.standard_normal(100) * np.mean(x_init)
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        else:
            x_init = rng.dirichlet(np.full(100, 0.05))
        _, pol_v, x = _hybrid_single_restart(x_init, 100, beta_arr, 15000, 200000)
        n += 1
        pool_100.append((pol_v, x.copy()))

    pool_100.sort(key=lambda t: t[0])

    # Phase 4: P=200 from top-3 P=100, ULTRA beta + heavy polish (50% budget)
    beta_ultra = np.array(BETA_ULTRA, dtype=np.float64)
    top3_100 = pool_100[:3]
    while time.time() - t0 < time_budget:
        if rng.random() < 0.8 and len(top3_100) > 0:
            idx = rng.integers(0, len(top3_100))
            x_init = upsample_solution(top3_100[idx][1], 100, P)
            x_init += rng.uniform(0.03, 0.15) * rng.standard_normal(P) * np.mean(x_init)
            x_init = np.maximum(x_init, 0.0)
            x_init /= x_init.sum()
        else:
            z = np.abs(rng.standard_cauchy(P))
            x_init = z / z.sum()
        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_ultra, 15000, 300000)
        n += 1
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


# ============================================================================
# P5: Hybrid Breeding + Warm Restart (combine session 1 & 2 winners)
# ============================================================================

def run_breed_and_warm(time_budget, seed=42):
    """
    Phase 1 (40%): Elite breeding with heavy-tailed inits to build pool
    Phase 2 (60%): Warm restarts from pool + best known, with adaptive perturbation
    """
    import glob
    rng = np.random.default_rng(seed)
    beta_arr = np.array(BETA_ULTRA, dtype=np.float64)

    # Load best known
    warm_x = load_best_solution()
    pool = [(verify_solution(warm_x), warm_x.copy())]

    best_val = pool[0][0]
    best_x = warm_x.copy()
    n = 0

    t0 = time.time()

    # Phase 1: Build pool with breeding
    while time.time() - t0 < time_budget * 0.4:
        n += 1
        if len(pool) >= 3 and rng.random() < 0.5:
            vals = np.array([p[0] for p in pool])
            probs = vals.max() - vals + 1e-8
            probs /= probs.sum()
            idx1, idx2 = rng.choice(len(pool), 2, replace=False, p=probs)
            alpha = rng.beta(2, 2)
            x_init = alpha * pool[idx1][1] + (1 - alpha) * pool[idx2][1]
            x_init += rng.standard_cauchy(P) * 0.015
        else:
            choice = rng.integers(0, 4)
            if choice == 0:
                x_init = rng.dirichlet(np.full(P, 0.05))
            elif choice == 1:
                z = np.abs(rng.standard_cauchy(P))
                x_init = z / z.sum()
            elif choice == 2:
                x_init = rng.random(P) ** rng.uniform(4, 12)
                x_init /= x_init.sum()
            else:
                x_init = rng.dirichlet(np.ones(P))

        x_init = np.maximum(x_init, 0.0)
        if x_init.sum() < 1e-12:
            x_init = rng.dirichlet(np.ones(P))
        else:
            x_init /= x_init.sum()

        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)

        if len(pool) < 10:
            pool.append((pol_v, x.copy()))
        else:
            worst = max(range(len(pool)), key=lambda i: pool[i][0])
            if pol_v < pool[worst][0]:
                pool[worst] = (pol_v, x.copy())

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    # Phase 2: Warm restarts from pool
    while time.time() - t0 < time_budget:
        n += 1
        # Pick from pool (favor best)
        vals = np.array([p[0] for p in pool])
        probs = vals.max() - vals + 1e-8
        probs /= probs.sum()
        idx = rng.choice(len(pool), p=probs)

        scale = rng.uniform(0.05, 0.5)
        x_init = pool[idx][1] + scale * rng.standard_normal(P) * np.mean(pool[idx][1])
        x_init = np.maximum(x_init, 0.0)
        if x_init.sum() < 1e-12:
            x_init = rng.dirichlet(np.ones(P))
        else:
            x_init /= x_init.sum()

        _, pol_v, x = _hybrid_single_restart(x_init, P, beta_arr, 15000, 200000)

        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()
            # Update pool
            worst = max(range(len(pool)), key=lambda i: pool[i][0])
            pool[worst] = (pol_v, x.copy())

    elapsed = time.time() - t0
    exact = verify_solution(best_x)
    return exact, best_x, n, elapsed


ALL = {
    "super_polish": ("Super Polish (2M iters)", run_super_polish),
    "extreme_sparse": ("Extreme Sparse Init", run_extreme_sparse),
    "multi_warm_polish": ("Multi-Warm Heavy Polish", run_multi_warm_heavy_polish),
    "basin_hunt": ("Basin Hunt + Upsample", run_basin_hunt_upsample),
    "breed_and_warm": ("Breed + Warm Restart", run_breed_and_warm),
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--n_trials", type=int, default=3)
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
        method_keys = list(ALL.keys())
    else:
        method_keys = args.methods.split(",")

    print(f"P={P}, budget={args.budget}s, trials={args.n_trials}")
    print(f"Methods: {method_keys}\n")

    results = {}
    for key in method_keys:
        name, func = ALL[key]
        print(f"\n{'-' * 60}")
        print(f"Method: {name}")
        print(f"{'-' * 60}")

        trials = []
        for trial in range(args.n_trials):
            seed = 42 + trial * 1000
            print(f"  Trial {trial+1}/{args.n_trials}...", end=" ", flush=True)
            val, x, restarts, elapsed = func(args.budget, seed=seed)
            trials.append({"value": val, "restarts": restarts, "elapsed": elapsed})
            save_solution(f"{key}_t{trial}", x, val)
            print(f"val={val:.8f}  restarts={restarts}  time={elapsed:.1f}s")

        vals = [t["value"] for t in trials]
        results[key] = {
            "best": min(vals),
            "mean": np.mean(vals),
            "std": np.std(vals) if len(vals) > 1 else 0,
            "trials": trials
        }
        print(f"  => best={min(vals):.8f} mean={np.mean(vals):.8f}")

    print(f"\n{'=' * 70}")
    print(f"PUSH BELOW SUMMARY (P={P})")
    print(f"{'=' * 70}")
    print(f"Target to beat: 1.51035713")
    print(f"{'Method':<30s} {'Best':>12s} {'Mean':>12s} {'Std':>10s}")
    print(f"{'-' * 30} {'-' * 12} {'-' * 12} {'-' * 10}")
    for key in results:
        name = ALL[key][0]
        r = results[key]
        print(f"{name:<30s} {r['best']:12.8f} {r['mean']:12.8f} {r['std']:10.8f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "push_below_data.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
