"""
sidon_cloud_fast.py — Modal cloud runner for Sidon autocorrelation optimization.

Runs warm-start + cold-start methods with progressive upsampling
(P=200 -> 500 -> 750 -> 1000) and cross-pollination. Uses exact Polyak polish
and FFT dispatch for P >= 300.

Setup:
    pip install modal
    modal setup

Usage:
    modal run sidon_cloud_fast.py --test           # quick validation (~1 min)
    modal run sidon_cloud_fast.py                  # full run, progressive to P=1000
    modal run --detach sidon_cloud_fast.py          # full run, survives laptop off
    modal run sidon_cloud_fast.py --p 500          # target grid size
    modal run sidon_cloud_fast.py --restarts 500   # custom restarts at final stage

Monitor:
    modal app logs sidon-fast-optimizer
    modal volume ls sidon-fast-results

Download results:
    modal volume get sidon-fast-results ./cloud_fast_results/ --force
"""

import modal
import json
import time
from pathlib import Path

# =============================================================================
# Modal configuration
# =============================================================================

CPU_CORES = 32
VOLUME_PATH = "/results"

app = modal.App("sidon-fast-optimizer")
volume = modal.Volume.from_name("sidon-fast-results", create_if_missing=True)

# Beta schedules — matching sidon_cloud.py for proper global exploration
BETA_HEAVY = [1, 1.5, 2, 3, 5, 8, 12, 18, 28, 42, 65, 100, 150, 230, 350,
              500, 750, 1000, 1500, 2000, 3000]

BETA_ULTRA = [1, 1.3, 1.7, 2.2, 3, 4, 5.5, 7.5, 10, 14, 20, 28, 40, 55,
              75, 100, 140, 200, 280, 400, 560, 800, 1100, 1500, 2000, 3000, 4000]

# Warm-start schedule: starts at beta=10 (not 42!) to preserve some global exploration
BETA_WARM = [10, 20, 40, 75, 150, 300, 600, 1000, 2000, 4000]


def _warmup_numba():
    """Pre-compile all Numba functions during image build."""
    from sidon_core import warmup
    warmup()
    print("Numba warmup complete.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2.0", "numba>=0.63", "joblib>=1.4")
    .add_local_python_source("sidon_core", copy=True)
    .run_function(_warmup_numba)
)


# =============================================================================
# Init generators — produce N initializations for each method
# =============================================================================

def _make_iterated_warm_inits(warm_x, P, n_restarts, rng):
    """Generate perturbations of warm_x at varying scales."""
    import numpy as np
    scales = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]
    inits = []
    for i in range(n_restarts):
        scale = scales[i % len(scales)]
        x = warm_x + scale * rng.standard_normal(P) * np.mean(warm_x)
        x = np.maximum(x, 0.0)
        if x.sum() < 1e-12:
            x = rng.dirichlet(np.ones(P))
        else:
            x /= x.sum()
        inits.append(x)
    return inits


def _make_adaptive_inits(warm_x, P, n_restarts, rng):
    """Generate perturbations at log-spaced scales."""
    import numpy as np
    scales = np.logspace(-2, 0.3, 8)
    inits = []
    for i in range(n_restarts):
        scale = scales[i % len(scales)]
        x = warm_x + scale * rng.standard_normal(P) * np.mean(warm_x)
        x = np.maximum(x, 0.0)
        if x.sum() < 1e-12:
            x = rng.dirichlet(np.ones(P))
        else:
            x /= x.sum()
        inits.append(x)
    return inits


def _make_cold_inits(strategy, P, n_restarts, rng):
    """Generate cold-start initializations (no warm start needed)."""
    import numpy as np
    centers = np.linspace(-0.25 + 0.25 / P, 0.25 - 0.25 / P, P)
    inits = []
    for _ in range(n_restarts):
        if strategy == "dirichlet_uniform":
            x = rng.dirichlet(np.ones(P))
        elif strategy == "symmetric_dirichlet":
            half = P // 2
            x_half = rng.dirichlet(np.ones(half))
            x = np.zeros(P)
            x[:half] = x_half
            x[P - half:] = x_half[::-1]
            if P % 2 == 1:
                x[half] = rng.uniform(0.0, 0.1)
            x /= x.sum()
        elif strategy == "gaussian_peak":
            sigma = rng.uniform(0.03, 0.15)
            mu = rng.uniform(-0.05, 0.05)
            x = np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
            x += rng.uniform(0, 0.01, P)
            x /= x.sum()
        elif strategy == "dirichlet_sparse":
            x = rng.dirichlet(np.full(P, 0.1))
        else:
            x = rng.dirichlet(np.ones(P))
        inits.append(x)
    return inits


# =============================================================================
# Modal remote function — runs one method
# =============================================================================

@app.function(
    cpu=CPU_CORES,
    memory=8192,
    timeout=86400,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def run_method(config: dict) -> dict:
    """
    Run one method on cloud CPUs.

    Uses FFT dispatch for P >= 300 (O(P log P) vs O(P^2)) and exact Polyak
    polish with per-iteration k_star tracking.

    Config keys:
        method, P, n_restarts, n_iters_lse, n_iters_polyak,
        beta_schedule, warm_x (list or None), seed, run_name
    """
    import numpy as np
    from sidon_core import autoconv_coeffs, hybrid_single_restart_dispatch
    from joblib import Parallel, delayed

    method = config["method"]
    P = config["P"]
    n_restarts = config.get("n_restarts", 300)
    n_iters_lse = config.get("n_iters_lse", 15000)
    n_iters_polyak = config.get("n_iters_polyak", 200000)
    beta_schedule = config.get("beta_schedule", BETA_WARM)
    seed = config.get("seed", 42)
    warm_x = np.array(config["warm_x"]) if config.get("warm_x") else None
    run_name = config.get("run_name", "fast")
    is_cold = config.get("is_cold", False)

    rng = np.random.default_rng(seed)
    beta_arr = np.array(beta_schedule, dtype=np.float64)

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{run_name}] START method={method} P={P} "
          f"restarts={n_restarts} cold={is_cold} cores={CPU_CORES}")

    # Generate initializations
    if is_cold:
        inits = _make_cold_inits(method, P, n_restarts, rng)
    elif method == "iterated_warm":
        inits = _make_iterated_warm_inits(warm_x, P, n_restarts, rng)
    elif method == "adaptive_perturbation":
        inits = _make_adaptive_inits(warm_x, P, n_restarts, rng)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Use FFT dispatch for large P, Numba for small P
    # hybrid_single_restart_dispatch auto-selects at P >= 300
    def _run_one(x_init):
        return hybrid_single_restart_dispatch(
            x_init, P, beta_arr, n_iters_lse, n_iters_polyak
        )

    # Run all restarts in parallel
    t0 = time.time()
    results = Parallel(n_jobs=CPU_CORES, verbose=0)(
        delayed(_run_one)(inits[i])
        for i in range(n_restarts)
    )

    # Find best
    best_val = np.inf
    best_x = None
    all_vals = []
    for lse_v, pol_v, x in results:
        all_vals.append(float(pol_v))
        if pol_v < best_val:
            best_val = float(pol_v)
            best_x = x.copy()

    # Exact verification
    exact_peak = float(np.max(autoconv_coeffs(best_x, P)))
    dt = time.time() - t0

    all_arr = np.array(all_vals)
    result = {
        "method": method,
        "P": P,
        "run_name": run_name,
        "exact_peak": exact_peak,
        "approx_peak": best_val,
        "median_peak": float(np.median(all_arr)),
        "std_peak": float(np.std(all_arr)),
        "min_peak": float(np.min(all_arr)),
        "n_restarts": n_restarts,
        "n_iters_lse": n_iters_lse,
        "n_iters_polyak": n_iters_polyak,
        "elapsed_s": dt,
        "simplex_weights": best_x.tolist(),
        "seed": seed,
        "cpu_cores": CPU_CORES,
        "is_cold": is_cold,
    }

    # Save to persistent volume
    fname = f"{run_name}_{method}_P{P}.json"
    out_path = Path(VOLUME_PATH) / fname
    with open(out_path, "w") as f:
        json.dump(result, f)
    volume.commit()

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{run_name}] DONE method={method} P={P} "
          f"exact={exact_peak:.6f} median={np.median(all_arr):.6f} "
          f"restarts={n_restarts} time={dt:.1f}s -> {fname}")

    return result


# =============================================================================
# Cross-pollination remote function
# =============================================================================

@app.function(
    cpu=CPU_CORES,
    memory=8192,
    timeout=86400,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def run_cross_pollination(config: dict) -> dict:
    """
    Cross-pollination: blend best solutions from multiple P values,
    add noise, then re-optimize. Explores space between known basins.
    """
    import numpy as np
    from sidon_core import (
        upsample_solution, hybrid_single_restart_dispatch, autoconv_coeffs,
    )
    from joblib import Parallel, delayed

    P_target = config["P_target"]
    solutions = config["solutions"]  # list of {"P": int, "x": list}
    n_blend = config.get("n_blend", 100)
    run_name = config.get("run_name", "xpoll")
    beta_schedule = np.array(
        config.get("beta_schedule", BETA_ULTRA), dtype=np.float64
    )
    n_iters_lse = config.get("n_iters_lse", 15000)
    n_iters_polyak = config.get("n_iters_polyak", 500000)

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{run_name}] Cross-pollination at P={P_target} "
          f"from {len(solutions)} sources, {n_blend} blends")

    t0 = time.time()

    # Upsample all source solutions to P_target
    upsampled = []
    for sol in solutions:
        P_src = sol["P"]
        x_src = np.array(sol["x"])
        if P_src != P_target:
            x_up = upsample_solution(x_src, P_src, P_target)
        else:
            x_up = x_src.copy()
        upsampled.append(x_up)

    # Generate blended initializations
    rng = np.random.default_rng()
    blend_inits = []
    for _ in range(n_blend):
        k = rng.integers(2, min(4, len(upsampled) + 1))
        idxs = rng.choice(len(upsampled), size=k, replace=False)
        weights = rng.dirichlet(np.ones(k))
        x_blend = sum(w * upsampled[i] for i, w in zip(idxs, weights))
        x_blend += 0.3 * rng.standard_normal(P_target) * np.mean(x_blend)
        x_blend = np.maximum(x_blend, 0.0)
        x_blend /= x_blend.sum()
        blend_inits.append(x_blend)

    # Run hybrid on all blended inits (FFT dispatch for large P)
    results = Parallel(n_jobs=CPU_CORES, verbose=0)(
        delayed(hybrid_single_restart_dispatch)(
            blend_inits[i], P_target, beta_schedule,
            n_iters_lse, n_iters_polyak
        )
        for i in range(n_blend)
    )

    best_val = np.inf
    best_x = None
    for lse_v, pol_v, x in results:
        if pol_v < best_val:
            best_val = pol_v
            best_x = x.copy()

    ev = float(np.max(autoconv_coeffs(best_x, P_target)))
    dt = time.time() - t0

    result = {
        "P": P_target,
        "method": "cross_pollination",
        "run_name": run_name,
        "exact_peak": ev,
        "approx_peak": float(best_val),
        "n_blend": n_blend,
        "elapsed_s": dt,
        "simplex_weights": best_x.tolist(),
        "cpu_cores": CPU_CORES,
    }

    fname = f"{run_name}_cross_pollination_P{P_target}.json"
    out_path = Path(VOLUME_PATH) / fname
    with open(out_path, "w") as f:
        json.dump(result, f)
    volume.commit()

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{run_name}] Cross-pollination P={P_target}: "
          f"exact={ev:.6f} time={dt:.1f}s")

    return result


# =============================================================================
# Helper to load warm-start solution
# =============================================================================

def load_warm_solution_local():
    """Load the best known solution from the experiments directory.

    Tries P=1000, P=500, P=200 in order, returns (x_list, source_P).
    """
    import numpy as np
    candidates = [
        ("experiments/solution_final_final_baseline_P1000.json", 1000),
        ("experiments/solution_final_final_baseline_P500.json", 500),
        ("experiments/solution_final_final_baseline_P200.json", 200),
    ]
    for fname, source_P in candidates:
        p = Path(fname)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return np.array(data["x"]).tolist(), source_P
    return None, None


def upsample_local(x_src, P_src, P_target):
    """Upsample a solution from P_src bins to P_target bins."""
    import numpy as np
    if P_src == P_target:
        return x_src
    x = np.array(x_src)
    edges_low = np.linspace(-0.25, 0.25, P_src + 1)
    edges_high = np.linspace(-0.25, 0.25, P_target + 1)
    bw_low = 0.5 / P_src
    bw_high = 0.5 / P_target
    h_low = x / bw_low
    c_low = 0.5 * (edges_low[:-1] + edges_low[1:])
    c_high = 0.5 * (edges_high[:-1] + edges_high[1:])
    h_high = np.interp(c_high, c_low, h_low)
    h_high = np.maximum(h_high, 0.0)
    x_high = h_high * bw_high
    if x_high.sum() > 0:
        x_high /= x_high.sum()
    else:
        x_high = np.ones(P_target) / P_target
    return x_high.tolist()


# =============================================================================
# Main entrypoint
# =============================================================================

# Warm-start methods (perturb existing best solution)
WARM_METHODS = ["iterated_warm", "adaptive_perturbation"]

# Cold-start methods (random init, explore new basins)
COLD_METHODS = ["dirichlet_uniform", "symmetric_dirichlet"]

ALL_METHODS = WARM_METHODS + COLD_METHODS

# Beta schedule per method type
METHOD_BETA = {
    # Warm methods: skip lowest betas but still start at 10 for some exploration
    "iterated_warm": BETA_WARM,
    "adaptive_perturbation": BETA_WARM,
    # Cold methods: full schedule starting at beta=1 for global exploration
    "dirichlet_uniform": BETA_HEAVY,
    "symmetric_dirichlet": BETA_HEAVY,
}

# Progressive upsampling stages
PROGRESSIVE_STAGES = [200, 500, 750, 1000]


def _polyak_iters_for_P(P):
    """Scale Polyak iterations with problem size."""
    if P <= 200:
        return 200000
    elif P <= 500:
        return 350000
    else:
        return 500000


@app.local_entrypoint()
def main(test: bool = False, p: int = 1000, restarts: int = 0):
    """
    Run warm + cold methods with progressive upsampling and cross-pollination.

    Progressive stages: P=200 -> 500 -> 750 -> 1000 (each warm-starts the next).
    Cold-start methods run in parallel to explore new basins.
    Final cross-pollination blends all discovered solutions.

    Args:
        test:     Quick validation (32 restarts, 1 method, 1 stage)
        p:        Target grid size (default 1000)
        restarts: Restarts per method at final stage (default 300)
    """
    t_total = time.time()
    P_final = p

    # Load initial warm-start solution
    warm_x, source_P = load_warm_solution_local()
    if warm_x is not None:
        print(f"Loaded warm-start solution (source P={source_P})")
    else:
        print("WARNING: No warm-start solution found. "
              "First stage will use random init.")

    if test:
        _run_test(P_final, warm_x, source_P)
    else:
        _run_full(P_final, warm_x, source_P, restarts)

    dt = time.time() - t_total
    print(f"\nTotal wall time: {dt:.0f}s ({dt/60:.1f} min)")
    print(f"\nDownload results:")
    print(f"  modal volume get sidon-fast-results ./cloud_fast_results/ --force")


def _run_test(P, warm_x, source_P):
    """Quick test: run 1 warm + 1 cold method with 32 restarts at target P."""
    print(f"\n{'='*70}")
    print(f"TEST MODE: iterated_warm + dirichlet_uniform, P={P}, 32 restarts")
    print(f"{'='*70}")

    if warm_x is not None and source_P != P:
        warm_x = upsample_local(warm_x, source_P, P)

    configs = []

    # Warm method
    warm_config = {
        "method": "iterated_warm",
        "P": P,
        "n_restarts": 32,
        "run_name": "test",
        "seed": 42,
        "beta_schedule": METHOD_BETA["iterated_warm"],
        "n_iters_lse": 15000,
        "n_iters_polyak": _polyak_iters_for_P(P),
        "is_cold": False,
    }
    if warm_x is not None:
        warm_config["warm_x"] = warm_x
    configs.append(warm_config)

    # Cold method
    configs.append({
        "method": "dirichlet_uniform",
        "P": P,
        "n_restarts": 32,
        "run_name": "test",
        "seed": 123,
        "beta_schedule": METHOD_BETA["dirichlet_uniform"],
        "n_iters_lse": 15000,
        "n_iters_polyak": _polyak_iters_for_P(P),
        "is_cold": True,
    })

    results = list(run_method.map(configs))
    best = min(results, key=lambda r: r["exact_peak"])
    for r in sorted(results, key=lambda r: r["exact_peak"]):
        marker = " ***" if r is best else ""
        print(f"  {r['method']:<25s} exact={r['exact_peak']:.6f} "
              f"median={r['median_peak']:.6f} time={r['elapsed_s']:.1f}s{marker}")


def _run_full(P_final, warm_x, source_P, restarts_override):
    """Progressive upsampling with warm + cold methods, then cross-pollination."""

    # Build stage list up to P_final
    stages = [p for p in PROGRESSIVE_STAGES if p <= P_final]
    if not stages or stages[-1] != P_final:
        stages.append(P_final)

    print(f"\n{'='*70}")
    print(f"PROGRESSIVE RUN: {' -> '.join(f'P={p}' for p in stages)}")
    print(f"  Warm methods:  {', '.join(WARM_METHODS)}")
    print(f"  Cold methods:  {', '.join(COLD_METHODS)}")
    print(f"  Cores/container: {CPU_CORES}")
    print(f"{'='*70}")

    best_x = warm_x
    best_source_P = source_P
    # Track best solution at each P for cross-pollination
    best_per_P = {}
    global_best_val = float("inf")
    global_best_result = None

    for stage_idx, P in enumerate(stages):
        is_final = (stage_idx == len(stages) - 1)

        # Upsample warm solution to this stage's P
        warm_x_P = None
        if best_x is not None:
            if best_source_P != P:
                warm_x_P = upsample_local(best_x, best_source_P, P)
            else:
                warm_x_P = best_x

        # Determine restarts
        if is_final and restarts_override > 0:
            n_restarts_warm = restarts_override
            n_restarts_cold = max(restarts_override // 2, 80)
        elif is_final:
            n_restarts_warm = 300
            n_restarts_cold = 150
        else:
            n_restarts_warm = 100
            n_restarts_cold = 60

        n_iters_polyak = _polyak_iters_for_P(P)

        print(f"\n--- Stage {stage_idx+1}/{len(stages)}: P={P} "
              f"({'FINAL' if is_final else 'intermediate'}) ---")
        print(f"    warm_restarts={n_restarts_warm}  cold_restarts={n_restarts_cold}  "
              f"polyak={n_iters_polyak}  warm={'yes' if warm_x_P is not None else 'no'}")

        # Build configs for all methods
        configs = []
        seed_counter = 42

        # Warm methods
        for method in WARM_METHODS:
            config = {
                "method": method,
                "P": P,
                "n_restarts": n_restarts_warm,
                "run_name": f"stage_P{P}",
                "seed": seed_counter,
                "beta_schedule": METHOD_BETA[method],
                "n_iters_lse": 15000,
                "n_iters_polyak": n_iters_polyak,
                "is_cold": False,
            }
            seed_counter += 1000
            if warm_x_P is not None:
                config["warm_x"] = warm_x_P
            else:
                # No warm start available — fall back to cold init
                config["is_cold"] = True
                config["beta_schedule"] = BETA_HEAVY
            configs.append(config)

        # Cold methods — explore new basins
        for method in COLD_METHODS:
            configs.append({
                "method": method,
                "P": P,
                "n_restarts": n_restarts_cold,
                "run_name": f"stage_P{P}",
                "seed": seed_counter,
                "beta_schedule": METHOD_BETA[method],
                "n_iters_lse": 15000,
                "n_iters_polyak": n_iters_polyak,
                "is_cold": True,
            })
            seed_counter += 1000

        # Launch all methods in parallel for this stage
        results = list(run_method.map(configs))

        # Find best result from this stage
        best_result = min(results, key=lambda r: r["exact_peak"])
        best_x = best_result["simplex_weights"]
        best_source_P = P

        # Track for cross-pollination
        best_per_P[str(P)] = {
            "val": best_result["exact_peak"],
            "x": best_result["simplex_weights"],
        }

        if best_result["exact_peak"] < global_best_val:
            global_best_val = best_result["exact_peak"]
            global_best_result = best_result

        # Print stage summary
        print(f"\n    {'Method':<25s} {'Cold':>5s} {'Exact':>10s} {'Median':>10s} "
              f"{'Std':>10s} {'Time':>7s}")
        print(f"    {'-'*25} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*7}")
        for r in sorted(results, key=lambda r: r["exact_peak"]):
            marker = " ***" if r["method"] == best_result["method"] else ""
            cold_str = "yes" if r.get("is_cold", False) else "no"
            print(f"    {r['method']:<25s} {cold_str:>5s} {r['exact_peak']:10.6f} "
                  f"{r['median_peak']:10.6f} {r['std_peak']:10.6f} "
                  f"{r['elapsed_s']:6.1f}s{marker}")
        print(f"    Stage best: {best_result['exact_peak']:.6f} "
              f"({best_result['method']})")

    # ── Cross-pollination ────────────────────────────────────────────
    solutions = [
        {"P": int(P_str), "x": best_per_P[P_str]["x"]}
        for P_str in best_per_P
    ]

    if len(solutions) >= 2:
        print(f"\n--- Cross-pollination ---")
        print(f"    Blending {len(solutions)} solutions at P={P_final}")

        xpoll_config = {
            "P_target": P_final,
            "solutions": solutions,
            "n_blend": 100,
            "run_name": "xpoll",
            "beta_schedule": BETA_ULTRA,
            "n_iters_lse": 15000,
            "n_iters_polyak": _polyak_iters_for_P(P_final),
        }
        xpoll_result = run_cross_pollination.remote(xpoll_config)

        print(f"    Cross-pollination: exact={xpoll_result['exact_peak']:.6f} "
              f"time={xpoll_result['elapsed_s']:.1f}s")

        if xpoll_result["exact_peak"] < global_best_val:
            global_best_val = xpoll_result["exact_peak"]
            global_best_result = xpoll_result
            best_x = xpoll_result["simplex_weights"]
            print(f"    *** NEW GLOBAL BEST from cross-pollination! ***")
    else:
        print("\n  Skipped cross-pollination: need at least 2 solutions")

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULT (P={P_final})")
    print(f"{'='*70}")
    print(f"BEST: {global_best_val:.6f} "
          f"({global_best_result['method'] if global_best_result else 'none'})")
    print(f"Literature best: 1.5029")
    if global_best_val < float("inf"):
        print(f"Gap: {global_best_val - 1.5029:+.6f}")

    for P_str in sorted(best_per_P, key=lambda k: int(k)):
        print(f"  P={P_str:>5}: {best_per_P[P_str]['val']:.6f}")

    # Save consolidated results
    summary = {
        "P_final": P_final,
        "stages": [p for p in stages],
        "best_method": global_best_result["method"] if global_best_result else None,
        "best_exact_peak": global_best_val,
        "best_simplex_weights": best_x if isinstance(best_x, list) else None,
        "best_per_P": {k: v["val"] for k, v in best_per_P.items()},
        "all_final_results": [
            {k: v for k, v in r.items() if k != "simplex_weights"}
            for r in results
        ],
    }
    fname = f"fast_summary_P{P_final}.json"
    _save_summary.remote(summary, fname)
    print(f"Saved summary to {fname}")


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=300,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def _save_summary(summary: dict, fname: str):
    """Save summary JSON to the volume."""
    out_path = Path(VOLUME_PATH) / fname
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
