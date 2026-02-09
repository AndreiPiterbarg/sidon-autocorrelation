"""
sidon_cloud.py — Modal.com cloud runner for Sidon autocorrelation optimization.

Runs the logsumexp_optimizer pipeline on cloud CPUs via Modal.
All core math lives in sidon_core.py (no Modal dependency there).


Setup:
    pip install modal
    modal setup

Usage:
    modal run sidon_cloud.py --test                # quick validation (~$0.50)
    modal run sidon_cloud.py                       # full 7-round pipeline
    modal run --detach sidon_cloud.py              # full run, survives laptop off
    modal run sidon_cloud.py --resume              # resume from checkpoint

Monitor:
    modal app logs sidon-optimizer                 # stream logs
    modal volume ls sidon-results                  # list saved files

Download results:
    modal volume get sidon-results ./cloud_results/ --force
"""

import modal
import json
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Modal configuration
# ═══════════════════════════════════════════════════════════════════════════════

CPU_CORES = 32
VOLUME_PATH = "/results"

app = modal.App("sidon-optimizer")
volume = modal.Volume.from_name("sidon-results", create_if_missing=True)


def _warmup_numba():
    """Pre-compile Numba functions during image build."""
    from sidon_core import warmup
    warmup()
    print("Numba warmup complete.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2.0", "numba>=0.63", "joblib>=1.4")
    .add_local_python_source("sidon_core", copy=True)
    .run_function(_warmup_numba)
)

# Beta schedules (also defined in sidon_core, duplicated here for config building)
BETA_HEAVY = [1, 1.5, 2, 3, 5, 8, 12, 18, 28, 42, 65, 100, 150, 230, 350,
              500, 750, 1000, 1500, 2000, 3000]

BETA_ULTRA = [1, 1.3, 1.7, 2.2, 3, 4, 5.5, 7.5, 10, 14, 20, 28, 40, 55,
              75, 100, 140, 200, 280, 400, 560, 800, 1100, 1500, 2000, 3000, 4000]

# Shorter schedule for warm_perturb: skip low-beta stages that just diffuse good warm starts
BETA_WARM = [10, 20, 40, 75, 150, 300, 600, 1000, 2000, 4000]

ALL_STRATEGIES = [
    'dirichlet_uniform', 'dirichlet_sparse', 'dirichlet_concentrated',
    'gaussian_peak', 'bimodal', 'cosine_shaped', 'triangle',
    'flat_noisy', 'boundary_heavy', 'random_sparse_k',
    'symmetric_dirichlet', 'warm_perturb',
]


# ═══════════════════════════════════════════════════════════════════════════════
# Modal remote functions
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(
    cpu=CPU_CORES,
    memory=8192,
    timeout=86400,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def run_strategy(config: dict) -> dict:
    """
    Run one strategy at one P value on cloud CPUs.

    Each invocation gets its own 32-core container. Within the container,
    joblib fans restarts across all cores.

    Config keys:
        P, strategy, n_restarts, n_iters_lse, n_iters_polyak,
        beta_schedule, warm_x (list or None), seed, round_name
    """
    import numpy as np
    from sidon_core import hybrid_strategy_run, autoconv_coeffs

    P = config["P"]
    strategy = config["strategy"]
    n_restarts = config.get("n_restarts", 80)
    n_iters_lse = config.get("n_iters_lse", 15000)
    n_iters_polyak = config.get("n_iters_polyak", 200000)
    beta_schedule = config.get("beta_schedule", BETA_HEAVY)
    seed = config.get("seed")
    warm_x = np.array(config["warm_x"]) if config.get("warm_x") else None
    round_name = config.get("round_name", "unknown")

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{round_name}] START P={P} strategy={strategy} "
          f"restarts={n_restarts} cores={CPU_CORES}")

    t0 = time.time()
    val, x, all_vals = hybrid_strategy_run(
        P, strategy, beta_schedule,
        n_iters_lse=n_iters_lse,
        n_iters_polyak=n_iters_polyak,
        n_restarts=n_restarts,
        n_jobs=CPU_CORES,
        warm_x=warm_x,
        seed=seed,
    )

    ev = float(np.max(autoconv_coeffs(x, P)))
    dt = time.time() - t0

    all_vals_arr = np.array(all_vals)
    result = {
        "P": P,
        "strategy": strategy,
        "round_name": round_name,
        "exact_peak": ev,
        "approx_peak": float(val),
        "median_peak": float(np.median(all_vals_arr)),
        "std_peak": float(np.std(all_vals_arr)),
        "n_restarts": n_restarts,
        "elapsed_s": dt,
        "simplex_weights": x.tolist(),
        "seed": seed,
        "cpu_cores": CPU_CORES,
    }

    # Save to persistent volume (compact JSON to reduce volume storage)
    fname = f"{round_name}_P{P}_{strategy}.json"
    out_path = Path(VOLUME_PATH) / fname
    with open(out_path, "w") as f:
        json.dump(result, f)
    volume.commit()

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{round_name}] DONE P={P} strategy={strategy} "
          f"exact={ev:.6f} approx={val:.6f} time={dt:.1f}s -> {fname}")

    return result


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
    from sidon_core import upsample_solution, _hybrid_single_restart, autoconv_coeffs
    from joblib import Parallel, delayed

    P_target = config["P_target"]
    solutions = config["solutions"]  # list of {"P": int, "x": list}
    n_blend = config.get("n_blend", 100)
    round_name = config.get("round_name", "r7")
    beta_schedule = np.array(
        config.get("beta_schedule", BETA_ULTRA), dtype=np.float64
    )

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{round_name}] Cross-pollination at P={P_target} "
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

    # Run hybrid on all blended inits
    results = Parallel(n_jobs=CPU_CORES, verbose=0)(
        delayed(_hybrid_single_restart)(
            blend_inits[i], P_target, beta_schedule, 15000, 500000
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
        "strategy": "cross_pollination",
        "round_name": round_name,
        "exact_peak": ev,
        "approx_peak": float(best_val),
        "n_blend": n_blend,
        "elapsed_s": dt,
        "simplex_weights": best_x.tolist(),
        "cpu_cores": CPU_CORES,
    }

    fname = f"{round_name}_P{P_target}_cross_pollination.json"
    out_path = Path(VOLUME_PATH) / fname
    with open(out_path, "w") as f:
        json.dump(result, f)
    volume.commit()

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{round_name}] Cross-pollination P={P_target}: "
          f"exact={ev:.6f} time={dt:.1f}s")

    return result


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=300,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def save_checkpoint(state: dict):
    """Save orchestration state to volume for resume."""
    out_path = Path(VOLUME_PATH) / "checkpoint.json"
    with open(out_path, "w") as f:
        json.dump(state, f, indent=2)
    volume.commit()

    gb = state["global_best"]
    print(f"Checkpoint saved: best={gb['val']:.6f} P={gb['P']} "
          f"strategy={gb['strategy']}")


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=300,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def load_checkpoint() -> dict:
    """Load orchestration state from volume. Returns None if no checkpoint."""
    volume.reload()
    checkpoint_path = Path(VOLUME_PATH) / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=300,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def save_final_results(global_best: dict, best_per_P: dict):
    """Save final consolidated results to volume."""
    import numpy as np

    save_data = {}
    for P_str, entry in best_per_P.items():
        P = int(P_str)
        x = entry["x"]
        edges = np.linspace(-0.25, 0.25, P + 1).tolist()
        bin_width = 0.5 / P
        heights = [xi / bin_width for xi in x]
        save_data[f"cloud_P{P}"] = {
            "P": P,
            "exact_peak": entry["val"],
            "simplex_weights": x,
            "edges": edges,
            "heights": heights,
        }

    save_data["global_best"] = {
        "P": global_best["P"],
        "exact_peak": global_best["val"],
        "strategy": global_best["strategy"],
        "round": global_best["round"],
        "simplex_weights": global_best["x"],
    }

    out_path = Path(VOLUME_PATH) / "best_solutions.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    volume.commit()
    print(f"Saved {len(save_data)} solutions to best_solutions.json on volume")


# ═══════════════════════════════════════════════════════════════════════════════
# Local helpers (run on your machine, not on Modal)
# ═══════════════════════════════════════════════════════════════════════════════

def update_best(global_best, best_per_P, result):
    """Update tracking state from a single strategy result."""
    P = result["P"]
    ev = result["exact_peak"]
    x = result["simplex_weights"]
    strategy = result["strategy"]
    round_name = result["round_name"]

    if ev < global_best["val"]:
        global_best["val"] = ev
        global_best["x"] = x
        global_best["P"] = P
        global_best["strategy"] = strategy
        global_best["round"] = round_name
        print(f"  *** NEW GLOBAL BEST: {ev:.6f} "
              f"(P={P}, {strategy}, {round_name}) ***")

    P_str = str(P)
    if P_str not in best_per_P or ev < best_per_P[P_str]["val"]:
        best_per_P[P_str] = {"val": ev, "x": x}


def make_config(round_name, P, strategy, n_restarts, beta_schedule,
                n_iters_lse=15000, n_iters_polyak=200000, warm_x=None):
    """Build config dict for run_strategy."""
    return {
        "round_name": round_name,
        "P": P,
        "strategy": strategy,
        "n_restarts": n_restarts,
        "n_iters_lse": n_iters_lse,
        "n_iters_polyak": n_iters_polyak,
        "beta_schedule": beta_schedule,
        "warm_x": warm_x,
    }


def upsample_local(best_per_P, target_P):
    """
    Upsample best available solution to target_P.
    Runs locally (numpy only, no sidon_core import needed).
    Returns list (for JSON serialization) or None.
    """
    import numpy as np

    # Find best source P < target_P, or any P if none smaller
    source_Ps = sorted(
        [int(p) for p in best_per_P],
        key=lambda p: abs(p - target_P)
    )
    if not source_Ps:
        return None

    source_P = source_Ps[0]
    x_src = np.array(best_per_P[str(source_P)]["x"])

    if source_P == target_P:
        return x_src.tolist()

    edges_low = np.linspace(-0.25, 0.25, source_P + 1)
    edges_high = np.linspace(-0.25, 0.25, target_P + 1)
    bw_low = 0.5 / source_P
    bw_high = 0.5 / target_P
    h_low = x_src / bw_low
    c_low = 0.5 * (edges_low[:-1] + edges_low[1:])
    c_high = 0.5 * (edges_high[:-1] + edges_high[1:])
    h_high = np.interp(c_high, c_low, h_low)
    h_high = np.maximum(h_high, 0.0)
    x_high = h_high * bw_high
    if x_high.sum() > 0:
        x_high /= x_high.sum()
    else:
        x_high = np.ones(target_P) / target_P
    return x_high.tolist()


def do_checkpoint(global_best, best_per_P, completed_round=None,
                   top6=None, top4=None):
    """Save checkpoint via remote call. Stores round progress + tournament state."""
    state = {
        "global_best": global_best,
        "best_per_P": best_per_P,
    }
    if completed_round is not None:
        state["completed_round"] = completed_round
    if top6 is not None:
        state["top6"] = top6
    if top4 is not None:
        state["top4"] = top4
    save_checkpoint.remote(state)

    # Also save consolidated best after every round (not just at end)
    if global_best["x"] is not None:
        save_final_results.remote(global_best, best_per_P)


def print_round_header(name, description):
    print(f"\n{'='*70}")
    print(f"{name}: {description}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(test: bool = False, resume: bool = False):
    """
    Orchestrate the full 7-round optimization on Modal cloud CPUs.

    Args:
        test:   Quick validation run (P=50, 10 restarts, 2 strategies)
        resume: Resume from a previous checkpoint on the volume
    """
    t_total = time.time()

    # Initialize tracking state
    global_best = {
        "val": float("inf"), "x": None, "P": None,
        "strategy": None, "round": None,
    }
    best_per_P = {}  # keys are str(P) for JSON compat

    # Resume from checkpoint?
    checkpoint = None
    completed_round = 0
    top6 = None
    top4 = None

    if resume:
        checkpoint = load_checkpoint.remote()
        if checkpoint:
            global_best = checkpoint["global_best"]
            best_per_P = checkpoint["best_per_P"]
            completed_round = checkpoint.get("completed_round", 0)
            top6 = checkpoint.get("top6")
            top4 = checkpoint.get("top4")
            print(f"Resumed: best={global_best['val']:.6f} "
                  f"P={global_best['P']} ({len(best_per_P)} P values)")
            if completed_round > 0:
                print(f"Skipping rounds 1-{completed_round} (already completed)")
        else:
            print("No checkpoint found, starting fresh.")

    if test:
        _run_test(global_best, best_per_P)
    else:
        _run_full(global_best, best_per_P, start_round=completed_round + 1,
                   top6=top6, top4=top4)

    # Final summary
    dt = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"COMPLETE — Wall time: {dt:.0f}s ({dt/60:.1f} min, {dt/3600:.2f} hr)")
    print(f"{'='*70}")
    print(f"GLOBAL BEST: {global_best['val']:.6f}")
    print(f"  P        = {global_best['P']}")
    print(f"  Strategy = {global_best['strategy']}")
    print(f"  Round    = {global_best['round']}")
    print(f"  Literature best: 1.5029")
    if global_best['val'] < float("inf"):
        print(f"  Gap:     {global_best['val'] - 1.5029:+.6f}")
    print()
    for P_str in sorted(best_per_P, key=lambda k: int(k)):
        print(f"  P={P_str:>5}: {best_per_P[P_str]['val']:.6f}")

    # Final save (do_checkpoint now also calls save_final_results)
    do_checkpoint(global_best, best_per_P, completed_round=9)

    print(f"\nDownload results:")
    print(f"  modal volume get sidon-results ./cloud_results/ --force")


# ═══════════════════════════════════════════════════════════════════════════════
# Test mode
# ═══════════════════════════════════════════════════════════════════════════════

def _run_test(global_best, best_per_P):
    """
    Cost-estimation test: runs small versions of key workloads to measure
    real per-container time, then extrapolates to the full pipeline.
    """
    RATE_PER_CORE_HR = 0.192 / 8  # Modal: ~$0.024/core/hr (approx)
    rate_per_container_sec = CPU_CORES * RATE_PER_CORE_HR / 3600

    print_round_header("COST ESTIMATION TEST",
        "Running 4 micro-benchmarks to predict full pipeline cost")

    # ── Benchmark 1: P=200, 10 restarts, BETA_HEAVY ──────────────
    # Represents Round 1 containers (12 strategies x 80 restarts)
    configs = [
        make_config("bench", 200, "dirichlet_uniform", 10, BETA_HEAVY,
                     n_iters_lse=15000, n_iters_polyak=200000),
    ]

    # ── Benchmark 2: P=500, 10 restarts, BETA_HEAVY ──────────────
    # Represents Rounds 3-4 containers
    configs.append(
        make_config("bench", 500, "gaussian_peak", 10, BETA_HEAVY,
                     n_iters_lse=15000, n_iters_polyak=300000),
    )

    # ── Benchmark 3: P=750, 10 restarts, BETA_ULTRA ──────────────
    # Represents Rounds 5 containers
    configs.append(
        make_config("bench", 750, "dirichlet_uniform", 10, BETA_ULTRA,
                     n_iters_lse=20000, n_iters_polyak=500000),
    )

    # ── Benchmark 4: P=200, warm_perturb, 10 restarts, BETA_WARM ─
    # Represents warm_perturb containers (shorter beta schedule)
    configs.append(
        make_config("bench", 200, "warm_perturb", 10, BETA_WARM,
                     n_iters_lse=15000, n_iters_polyak=200000),
    )

    print(f"  Launching 4 benchmark containers ({CPU_CORES} cores each)...")
    results = list(run_strategy.map(configs))

    # Parse timing
    bench = {}
    for r in results:
        key = f"P{r['P']}_{r['strategy']}"
        bench[key] = r
        print(f"  {key:40s}  time={r['elapsed_s']:.1f}s  peak={r['exact_peak']:.6f}")
        update_best(global_best, best_per_P, r)

    # Extrapolate: scale by (full_restarts / bench_restarts)
    # Time scales roughly linearly with restarts (joblib saturates cores)
    t200 = bench["P200_dirichlet_uniform"]["elapsed_s"]
    t500 = bench["P500_gaussian_peak"]["elapsed_s"]
    t750 = bench["P750_dirichlet_uniform"]["elapsed_s"]
    t_warm = bench["P200_warm_perturb"]["elapsed_s"]

    # Scale factor for restarts: full uses 60-150, bench used 10
    # Also scale for polyak iters where different
    def scale(bench_time, bench_restarts, full_restarts):
        return bench_time * (full_restarts / bench_restarts)

    # Round 1: 12 containers x P=200 x 80 restarts
    r1_per = scale(t200, 10, 80)
    r1_cost = 12 * r1_per * rate_per_container_sec

    # Round 2: ~7 containers x P=300 x 100 restarts
    r2_per = scale(t200, 10, 100) * 1.5  # P=300 ~1.5x slower than P=200
    r2_cost = 7 * r2_per * rate_per_container_sec

    # Round 3: ~6 containers x P=500 x 80 restarts
    r3_per = scale(t500, 10, 80)
    r3_cost = 6 * r3_per * rate_per_container_sec

    # Round 4: 2 containers x P=750 (120 + 60 restarts)
    r4_cost = (scale(t750, 10, 120) + scale(t750, 10, 60)) * \
              rate_per_container_sec

    # Round 5: ~5 containers x P=750 (200 warm + 4x80 cold)
    r5_cost = (scale(t750, 10, 200) + 4 * scale(t750, 10, 80)) * \
              rate_per_container_sec

    # Round 6: ~5 containers x P=1000 (160 warm + 4x80 cold)
    t1000_est = t750 * 2.0  # P=1000 roughly 2x slower than P=750
    r6_cost = (scale(t1000_est, 10, 160) + 4 * scale(t1000_est, 10, 80)) * \
              rate_per_container_sec

    # Round 7: P=1000 warm refinement (150 warm + 60 cold)
    r7_cost = (scale(t1000_est, 10, 150) + scale(t1000_est, 10, 60)) * \
              rate_per_container_sec

    # Round 8: P=1500 exploration (60 warm + 30 cold)
    t1500_est = t750 * 4.5  # P=1500 roughly 4.5x slower than P=750
    r8_cost = (scale(t1500_est, 10, 60) + scale(t1500_est, 10, 30)) * \
              rate_per_container_sec

    # Round 9: Cross-pollination at P=750, 1000, 1500
    r9_cost = (scale(t750, 10, 100) + scale(t1000_est, 10, 100) +
               scale(t1500_est, 10, 90)) * rate_per_container_sec

    total_cost = (r1_cost + r2_cost + r3_cost + r4_cost + r5_cost +
                  r6_cost + r7_cost + r8_cost + r9_cost)

    # Benchmark cost (what we just spent)
    bench_cost = 4 * max(r["elapsed_s"] for r in results) * \
                 rate_per_container_sec

    print(f"\n{'='*70}")
    print(f"COST ESTIMATE (based on measured timings)")
    print(f"{'='*70}")
    print(f"  Rate: ~${rate_per_container_sec * 3600:.2f}/container-hr "
          f"({CPU_CORES} cores)")
    print(f"")
    print(f"  Round 1 (P=200, 12 strats x 80):      ${r1_cost:6.2f}  "
          f"  ~{r1_per/60:.0f} min/container")
    print(f"  Round 2 (P=300, ~7 strats x 100):      ${r2_cost:6.2f}")
    print(f"  Round 3 (P=500, ~6 strats x 80):       ${r3_cost:6.2f}  "
          f"  ~{r3_per/60:.0f} min/container")
    print(f"  Round 4 (P=750, 2 containers):         ${r4_cost:6.2f}")
    print(f"  Round 5 (P=750, ~5 strats, 200 warm):  ${r5_cost:6.2f}")
    print(f"  Round 6 (P=1000, ~5 strats, 160 warm): ${r6_cost:6.2f}")
    print(f"  Round 7 (P=1000, warm refinement):     ${r7_cost:6.2f}")
    print(f"  Round 8 (P=1500, exploration):         ${r8_cost:6.2f}")
    print(f"  Round 9 (cross-pollination, 3 targets):${r9_cost:6.2f}")
    print(f"  ────────────────────────────────────────────")
    print(f"  TOTAL ESTIMATED COST:                  ${total_cost:6.2f}")
    print(f"  This benchmark cost:                   ~${bench_cost:.2f}")
    print(f"")
    print(f"  Note: Estimates assume linear scaling with restarts.")
    print(f"  Actual cost may vary ~20% due to container startup,")
    print(f"  early stopping, and joblib scheduling overhead.")
    print(f"")
    print(f"  Best from benchmark: {global_best['val']:.6f}")
    print(f"  Run without --test for the full pipeline.")


# ═══════════════════════════════════════════════════════════════════════════════
# Full 7-round pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_full(global_best, best_per_P, start_round=1, top6=None, top4=None):
    """Full 7-round optimization with round-skipping for --resume."""

    # Default tournament winners if resuming past rounds 1-2
    if top6 is None:
        top6 = ['warm_perturb', 'dirichlet_uniform', 'gaussian_peak',
                'symmetric_dirichlet', 'dirichlet_sparse', 'bimodal']
    if top4 is None:
        top4 = ['warm_perturb', 'dirichlet_uniform', 'gaussian_peak',
                'symmetric_dirichlet']

    # ── Round 1: Strategy sweep at P=200 ──────────────────────────
    if start_round <= 1:
        print_round_header("ROUND 1",
            f"Strategy sweep at P=200 ({len(ALL_STRATEGIES)} strategies x 80 restarts)")

        r1_configs = [
            make_config("r1", 200, s, 80, BETA_HEAVY)
            for s in ALL_STRATEGIES
        ]
        r1_results = list(run_strategy.map(r1_configs))
        for r in r1_results:
            update_best(global_best, best_per_P, r)

        r1_ranked = sorted(r1_results, key=lambda r: r["exact_peak"])
        top6 = [r["strategy"] for r in r1_ranked[:6]]

        print(f"\nRound 1 ranking:")
        for i, r in enumerate(r1_ranked):
            marker = " <--" if r["strategy"] in top6 else ""
            print(f"  {i+1:>2}. {r['strategy']:30s}  {r['exact_peak']:.6f}{marker}")
        print(f"Top 6 advancing: {top6}")

        do_checkpoint(global_best, best_per_P, completed_round=1, top6=top6)

    # ── Round 2: Top strategies at P=300 ──────────────────────────
    if start_round <= 2:
        r2_strats = list(set(top6 + ["warm_perturb"]))
        print_round_header("ROUND 2",
            f"Top strategies at P=300 ({len(r2_strats)} strategies x 100 restarts)")

        warm_300 = upsample_local(best_per_P, 300)
        r2_configs = [
            make_config("r2", 300, s, 100,
                         BETA_WARM if s == "warm_perturb" else BETA_HEAVY,
                         n_iters_polyak=300000,
                         warm_x=warm_300 if s == "warm_perturb" else None)
            for s in r2_strats
        ]
        r2_results = list(run_strategy.map(r2_configs))
        for r in r2_results:
            update_best(global_best, best_per_P, r)

        r2_ranked = sorted(r2_results, key=lambda r: r["exact_peak"])
        top4 = [r["strategy"] for r in r2_ranked[:4]]
        print(f"Top 4 after Round 2: {top4}")

        do_checkpoint(global_best, best_per_P, completed_round=2,
                       top6=top6, top4=top4)

    # ── Round 3: Top strategies at P=500 ──────────────────────────
    if start_round <= 3:
        r3_strats = list(set(top4 + ["warm_perturb", "dirichlet_uniform"]))
        print_round_header("ROUND 3",
            f"Top strategies at P=500 ({len(r3_strats)} strategies x 80 restarts)")

        warm_500 = upsample_local(best_per_P, 500)
        r3_configs = [
            make_config("r3", 500, s, 80,
                         BETA_WARM if s == "warm_perturb" else BETA_HEAVY,
                         n_iters_polyak=300000,
                         warm_x=warm_500 if s == "warm_perturb" else None)
            for s in r3_strats
        ]
        r3_results = list(run_strategy.map(r3_configs))
        for r in r3_results:
            update_best(global_best, best_per_P, r)

        do_checkpoint(global_best, best_per_P, completed_round=3,
                       top6=top6, top4=top4)

    # ── Round 4: Warm-start cascade -> P=750 ─────────────────────
    if start_round <= 4:
        print_round_header("ROUND 4",
            "Warm-start cascade -> P=750")

        warm_x = upsample_local(best_per_P, 750)
        if warm_x is not None:
            r4_configs = [
                make_config("r4", 750, "warm_perturb", 120, BETA_WARM,
                             n_iters_lse=20000, n_iters_polyak=500000,
                             warm_x=warm_x),
                make_config("r4", 750, "dirichlet_uniform", 60, BETA_HEAVY,
                             n_iters_lse=20000, n_iters_polyak=500000),
            ]
            r4_results = list(run_strategy.map(r4_configs))
            for r in r4_results:
                update_best(global_best, best_per_P, r)

        do_checkpoint(global_best, best_per_P, completed_round=4,
                       top6=top6, top4=top4)

    # ── Round 5: Heavy compute at P=750 ──────────────────────────
    if start_round <= 5:
        print_round_header("ROUND 5",
            "Heavy compute at P=750")

        r5_strats = list(set(top4 + ["warm_perturb"]))
        r5_configs = []
        warm_x = upsample_local(best_per_P, 750)
        for strategy in r5_strats:
            n_restarts = 200 if strategy == "warm_perturb" else 80
            is_warm = strategy == "warm_perturb"
            r5_configs.append(
                make_config("r5", 750, strategy, n_restarts,
                             BETA_WARM if is_warm else BETA_ULTRA,
                             n_iters_lse=20000, n_iters_polyak=500000,
                             warm_x=warm_x if is_warm else None)
            )
        r5_results = list(run_strategy.map(r5_configs))
        for r in r5_results:
            update_best(global_best, best_per_P, r)

        do_checkpoint(global_best, best_per_P, completed_round=5,
                       top6=top6, top4=top4)

    # ── Round 6: First push at P=1000 ─────────────────────────────
    if start_round <= 6:
        print_round_header("ROUND 6", "First push at P=1000")

        warm_1000 = upsample_local(best_per_P, 1000)
        r6_strats = list(set(top4 + ["warm_perturb"]))
        r6_configs = []
        for strategy in r6_strats:
            n_restarts = 160 if strategy == "warm_perturb" else 80
            is_warm = strategy == "warm_perturb"
            r6_configs.append(
                make_config("r6", 1000, strategy, n_restarts,
                             BETA_WARM if is_warm else BETA_ULTRA,
                             n_iters_lse=15000, n_iters_polyak=500000,
                             warm_x=warm_1000 if is_warm else None)
            )
        r6_results = list(run_strategy.map(r6_configs))
        for r in r6_results:
            update_best(global_best, best_per_P, r)

        do_checkpoint(global_best, best_per_P, completed_round=6,
                       top6=top6, top4=top4)

    # ── Round 7: P=1000 warm-start refinement ─────────────────────
    #    Second pass using Round 6's best P=1000 solution as warm start.
    #    Fresh perturbations from a better basin often beat the first pass.
    if start_round <= 7:
        print_round_header("ROUND 7",
            "P=1000 warm-start refinement (2nd pass from R6 best)")

        warm_1000 = upsample_local(best_per_P, 1000)
        if warm_1000 is not None:
            r7_configs = [
                make_config("r7", 1000, "warm_perturb", 150, BETA_WARM,
                             n_iters_lse=15000, n_iters_polyak=500000,
                             warm_x=warm_1000),
                make_config("r7", 1000, "dirichlet_uniform", 60, BETA_ULTRA,
                             n_iters_lse=15000, n_iters_polyak=500000),
            ]
            r7_results = list(run_strategy.map(r7_configs))
            for r in r7_results:
                update_best(global_best, best_per_P, r)

        do_checkpoint(global_best, best_per_P, completed_round=7,
                       top6=top6, top4=top4)

    # ── Round 8: P=1500 exploration ───────────────────────────────
    #    Higher resolution = more degrees of freedom to minimize the peak.
    #    Literature best (1.5029) used 30,000 pieces; P=1500 is a step
    #    toward that regime.
    if start_round <= 8:
        print_round_header("ROUND 8",
            "P=1500 exploration (higher resolution)")

        warm_1500 = upsample_local(best_per_P, 1500)
        r8_configs = [
            make_config("r8", 1500, "warm_perturb", 60, BETA_WARM,
                         n_iters_lse=15000, n_iters_polyak=500000,
                         warm_x=warm_1500),
            make_config("r8", 1500, "dirichlet_uniform", 30, BETA_ULTRA,
                         n_iters_lse=15000, n_iters_polyak=500000),
        ]
        r8_results = list(run_strategy.map(r8_configs))
        for r in r8_results:
            update_best(global_best, best_per_P, r)

        do_checkpoint(global_best, best_per_P, completed_round=8,
                       top6=top6, top4=top4)

    # ── Round 9: Cross-pollination ────────────────────────────────
    #    Blend best solutions across P values, re-optimize.
    #    Now includes P=1500 for higher-resolution blends.
    if start_round <= 9:
        print_round_header("ROUND 9",
            "Cross-pollination at P=750, 1000, 1500")

        solutions = [
            {"P": int(P_str), "x": best_per_P[P_str]["x"]}
            for P_str in best_per_P
        ]

        if len(solutions) >= 2:
            r9_configs = [
                {
                    "P_target": P_target,
                    "solutions": solutions,
                    "n_blend": n_blend,
                    "round_name": "r9",
                    "beta_schedule": BETA_ULTRA,
                }
                for P_target, n_blend in [(750, 100), (1000, 100), (1500, 90)]
            ]
            r9_results = list(run_cross_pollination.map(r9_configs))
            for r in r9_results:
                update_best(global_best, best_per_P, r)
        else:
            print("  Skipped: need at least 2 solutions for cross-pollination")

        do_checkpoint(global_best, best_per_P, completed_round=9,
                       top6=top6, top4=top4)
