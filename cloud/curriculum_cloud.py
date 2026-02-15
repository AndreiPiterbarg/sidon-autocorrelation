"""
curriculum_cloud.py — Modal cloud runner for curriculum learning optimization.

Runs massive low-P exploration then upsamples via cascade to high-P.
Core math in sidon_core.py, curriculum helpers in curriculum_core.py.

Setup:
    pip install modal
    modal setup

Usage:
    modal run curriculum_cloud.py --test           # cost estimation (~$0.30)
    modal run curriculum_cloud.py                  # full pipeline
    modal run --detach curriculum_cloud.py         # detached (survives laptop off)
    modal run curriculum_cloud.py --resume         # resume from checkpoint

Monitor:
    modal app logs curriculum-optimizer
    modal volume ls sidon-results

Download:
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

app = modal.App("curriculum-optimizer")
volume = modal.Volume.from_name("sidon-results", create_if_missing=True)


def _warmup_numba():
    """Pre-compile Numba functions during image build."""
    from curriculum_core import warmup
    warmup()
    print("Numba warmup complete.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=2.0", "numba>=0.63", "joblib>=1.4", "scipy>=1.14")
    .add_local_python_source("sidon_core", copy=True)
    .add_local_python_source("curriculum_core", copy=True)
    .run_function(_warmup_numba)
)

# Beta schedules (duplicated from sidon_core for config building)
BETA_HEAVY = [1, 1.5, 2, 3, 5, 8, 12, 18, 28, 42, 65, 100, 150, 230, 350,
              500, 750, 1000, 1500, 2000, 3000]

BETA_ULTRA = [1, 1.3, 1.7, 2.2, 3, 4, 5.5, 7.5, 10, 14, 20, 28, 40, 55,
              75, 100, 140, 200, 280, 400, 560, 800, 1100, 1500, 2000, 3000, 4000]

CURRICULUM_STRATEGIES = [
    'dirichlet_sparse', 'gaussian_peak', 'bimodal',
    'boundary_heavy', 'symmetric_dirichlet', 'warm_perturb',
]


# ═══════════════════════════════════════════════════════════════════════════════
# Remote functions
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(
    cpu=CPU_CORES,
    memory=8192,
    timeout=86400,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def run_explore(config: dict) -> dict:
    """
    Explore one P value: all strategies x n_restarts.
    One container per P value; restarts parallelized across 32 cores.
    """
    from curriculum_core import explore_single_P

    P = config["P"]
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] START explore P={P}")
    t0 = time.time()

    result = explore_single_P(
        P=P,
        strategies=config.get("strategies", CURRICULUM_STRATEGIES),
        n_restarts_per_strategy=config.get("n_restarts_per_strategy", 5000),
        beta_schedule=config.get("beta_schedule", BETA_HEAVY),
        n_iters_lse=config.get("n_iters_lse", 10000),
        n_iters_polyak=config.get("n_iters_polyak", 200000),
        diversity_threshold=config.get("diversity_threshold", 0.1),
        top_k=config.get("top_k", 10),
        n_jobs=CPU_CORES,
        seed=config.get("seed"),
    )
    result["elapsed_s"] = time.time() - t0

    # Save to persistent volume
    fname = f"curriculum_explore_P{P}.json"
    with open(Path(VOLUME_PATH) / fname, "w") as f:
        json.dump(result, f)
    volume.commit()

    # Shut down loky executor to avoid lingering background thread
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] DONE explore P={P}: best={result['best_val']:.6f} "
          f"diverse={len(result['diverse'])} time={result['elapsed_s']:.1f}s")

    return result


@app.function(
    cpu=CPU_CORES,
    memory=32768,
    timeout=86400,
    volumes={VOLUME_PATH: volume},
    image=image,
)
def run_cascade(config: dict) -> dict:
    """
    Optimize one upsampled candidate at a cascade level.
    Direct polish + warm perturbations, parallelized across 32 cores.
    """
    import numpy as np
    from curriculum_core import cascade_candidate

    P_target = config["P_target"]
    cand_idx = config.get("cand_idx", 0)
    x_up = np.array(config["x_up"])

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] START cascade P={P_target} candidate #{cand_idx}")
    t0 = time.time()

    results = cascade_candidate(
        x_up=x_up,
        P_target=P_target,
        n_warm_restarts=config.get("n_warm_restarts", 50),
        beta_schedule_direct=config.get("beta_schedule_direct", BETA_ULTRA),
        beta_schedule_warm=config.get("beta_schedule_warm", BETA_HEAVY),
        n_iters_lse=config.get("n_iters_lse", 10000),
        n_iters_polyak=config.get("n_iters_polyak", 500000),
        n_jobs=CPU_CORES,
        seed=config.get("seed"),
    )

    dt = time.time() - t0
    best_val = min(r[0] for r in results)

    # Save individual result
    fname = f"curriculum_cascade_P{P_target}_cand{cand_idx}.json"
    save_data = {"P_target": P_target, "cand_idx": cand_idx,
                 "best_val": best_val, "elapsed_s": dt}
    with open(Path(VOLUME_PATH) / fname, "w") as f:
        json.dump(save_data, f)
    volume.commit()

    # Shut down loky executor to avoid lingering background thread
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True)

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] DONE cascade P={P_target} cand #{cand_idx}: "
          f"best={best_val:.6f} n={len(results)} time={dt:.1f}s")

    return {
        "P_target": P_target,
        "cand_idx": cand_idx,
        "results": results,
        "best_val": best_val,
        "elapsed_s": dt,
    }


@app.function(
    cpu=1.0, memory=2048, timeout=300,
    volumes={VOLUME_PATH: volume}, image=image,
)
def save_checkpoint(state: dict):
    """Save orchestration state to volume for resume."""
    with open(Path(VOLUME_PATH) / "curriculum_checkpoint.json", "w") as f:
        json.dump(state, f, indent=2)
    volume.commit()

    stage = state.get("completed_stage", "?")
    gb = state.get("global_best", {})
    val = gb.get("val", "?")
    print(f"Checkpoint: stage={stage} best={val}")


@app.function(
    cpu=1.0, memory=2048, timeout=300,
    volumes={VOLUME_PATH: volume}, image=image,
)
def load_checkpoint() -> dict:
    """Load checkpoint from volume. Returns None if no checkpoint."""
    volume.reload()
    path = Path(VOLUME_PATH) / "curriculum_checkpoint.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@app.function(
    cpu=1.0, memory=2048, timeout=300,
    volumes={VOLUME_PATH: volume}, image=image,
)
def save_final_results(global_best: dict, best_per_P: dict):
    """Save consolidated results to volume."""
    import numpy as np

    save_data = {}
    for P_str, entry in best_per_P.items():
        P = int(P_str)
        x = entry["x"]
        edges = np.linspace(-0.25, 0.25, P + 1).tolist()
        bin_width = 0.5 / P
        heights = [xi / bin_width for xi in x]
        save_data[f"curriculum_P{P}"] = {
            "method": "curriculum_learning",
            "P": P,
            "exact_peak": entry["val"],
            "simplex_weights": x,
            "edges": edges,
            "heights": heights,
        }

    save_data["global_best"] = {
        "P": global_best["P"],
        "exact_peak": global_best["val"],
        "strategy": "curriculum_learning",
        "simplex_weights": global_best["x"],
    }

    with open(Path(VOLUME_PATH) / "curriculum_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    volume.commit()
    print(f"Saved {len(save_data)} entries to curriculum_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Local helpers (run on your machine, not on Modal)
# ═══════════════════════════════════════════════════════════════════════════════

def upsample_local(x_src, P_src, P_target):
    """Upsample via linear interpolation of heights (no sidon_core needed)."""
    import numpy as np

    x = np.array(x_src)
    if P_src == P_target:
        return x.tolist()

    edges_low = np.linspace(-0.25, 0.25, P_src + 1)
    edges_high = np.linspace(-0.25, 0.25, P_target + 1)
    bw_low, bw_high = 0.5 / P_src, 0.5 / P_target

    h_low = x / bw_low
    c_low = 0.5 * (edges_low[:-1] + edges_low[1:])
    c_high = 0.5 * (edges_high[:-1] + edges_high[1:])

    h_high = np.interp(c_high, c_low, h_low)
    h_high = np.maximum(h_high, 0.0)
    x_high = h_high * bw_high
    x_high = x_high / x_high.sum() if x_high.sum() > 0 else np.ones(P_target) / P_target
    return x_high.tolist()


def keep_diverse_local(solutions, threshold=0.1, max_keep=10):
    """Diversity filter (runs locally, solutions have x as lists)."""
    import numpy as np

    if not solutions:
        return []
    kept = [solutions[0]]
    for val, x in solutions[1:]:
        if len(kept) >= max_keep:
            break
        x_arr = np.array(x)
        if all(np.linalg.norm(x_arr - np.array(xk)) >= threshold for _, xk in kept):
            kept.append((val, x))
    return kept


def update_best(global_best, best_per_P, val, x, P):
    """Update tracking state from a single result."""
    if val < global_best["val"]:
        global_best["val"] = val
        global_best["x"] = x
        global_best["P"] = P
        print(f"  *** NEW GLOBAL BEST: {val:.6f} (P={P}) ***")
    P_str = str(P)
    if P_str not in best_per_P or val < best_per_P[P_str]["val"]:
        best_per_P[P_str] = {"val": val, "x": x}


def do_checkpoint(global_best, best_per_P, completed_stage, explore_diverse=None):
    """Save checkpoint + final results via remote calls."""
    state = {
        "global_best": global_best,
        "best_per_P": best_per_P,
        "completed_stage": completed_stage,
    }
    if explore_diverse is not None:
        state["explore_diverse"] = explore_diverse
    save_checkpoint.remote(state)

    if global_best["x"] is not None:
        save_final_results.remote(global_best, best_per_P)


def print_header(name, desc):
    print(f"\n{'='*70}")
    print(f"{name}: {desc}")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline orchestrator — runs ON Modal (survives laptop disconnect)
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(
    cpu=1.0,
    memory=4096,
    timeout=86400,  # 24 hours (Modal max)
    volumes={VOLUME_PATH: volume},
    image=image,
)
def run_pipeline(test: bool = False, resume: bool = False) -> dict:
    """
    Full pipeline orchestration running on Modal's infrastructure.
    Spawns run_explore/run_cascade containers via nested .map()/.remote() calls.
    With --detach, this function stays alive even if your laptop disconnects.
    """
    t_total = time.time()

    global_best = {
        "val": float("inf"), "x": None, "P": None,
    }
    best_per_P = {}
    explore_diverse = {}  # str(P) -> [(val, x_list), ...]
    completed_stage = None

    if resume:
        cp = load_checkpoint.remote()
        if cp:
            global_best = cp["global_best"]
            best_per_P = cp["best_per_P"]
            completed_stage = cp.get("completed_stage")
            explore_diverse = cp.get("explore_diverse", {})
            print(f"Resumed: best={global_best['val']:.6f} stage={completed_stage}")
        else:
            print("No checkpoint found, starting fresh.")

    if test:
        _run_test(global_best, best_per_P)
    else:
        _run_full(global_best, best_per_P, explore_diverse, completed_stage)

    # Final summary
    dt = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"COMPLETE — Wall time: {dt:.0f}s ({dt/60:.1f} min, {dt/3600:.2f} hr)")
    print(f"{'='*70}")
    if global_best["val"] < float("inf"):
        print(f"GLOBAL BEST: {global_best['val']:.6f} at P={global_best['P']}")
        print(f"  AE25 reference:  1.5032")
        print(f"  TTT26 reference: 1.50286")
        print(f"  Gap vs AE25:     {global_best['val'] - 1.5032:+.6f}")
    for P_str in sorted(best_per_P, key=lambda k: int(k)):
        print(f"  P={P_str:>5}: {best_per_P[P_str]['val']:.6f}")

    do_checkpoint(global_best, best_per_P, "done", explore_diverse)

    return {
        "global_best": global_best,
        "best_per_P": {k: v["val"] for k, v in best_per_P.items()},
    }


@app.local_entrypoint()
def main(test: bool = False, resume: bool = False):
    """Thin local wrapper — kicks off run_pipeline on Modal, then prints results."""
    print("Launching pipeline on Modal cloud...")
    print("  Safe to close laptop if using: modal run --detach curriculum_cloud.py")
    print("  Monitor with: modal app logs curriculum-optimizer\n")

    result = run_pipeline.remote(test=test, resume=resume)

    if result and result.get("global_best", {}).get("val", float("inf")) < float("inf"):
        gb = result["global_best"]
        print(f"\nGLOBAL BEST: {gb['val']:.6f} at P={gb['P']}")
    print(f"\nDownload: modal volume get sidon-results ./cloud_results/ --force")


# ═══════════════════════════════════════════════════════════════════════════════
# Test mode — cost estimation
# ═══════════════════════════════════════════════════════════════════════════════

def _run_test(global_best, best_per_P):
    """Run mini benchmarks and extrapolate full pipeline cost."""
    RATE_PER_CORE_HR = 0.192 / 8  # ~$0.024/core/hr on Modal
    rate_per_container_sec = CPU_CORES * RATE_PER_CORE_HR / 3600

    print_header("COST ESTIMATION TEST",
                 "Running micro-benchmarks to predict full pipeline cost")

    # ── Benchmark 1: Explore P=30, 10 restarts per strategy ────────
    explore_config = {
        "P": 30,
        "n_restarts_per_strategy": 10,
        "beta_schedule": BETA_HEAVY,
        "n_iters_lse": 10000,
        "n_iters_polyak": 200000,
        "diversity_threshold": 0.1,
        "top_k": 5,
        "seed": 42,
    }

    # ── Benchmark 2: Cascade P=100, 1 candidate, 5 warm restarts ──
    x_init = [1.0 / 30] * 30
    cascade_config = {
        "P_target": 100,
        "cand_idx": 0,
        "x_up": upsample_local(x_init, 30, 100),
        "n_warm_restarts": 5,
        "beta_schedule_direct": BETA_ULTRA,
        "beta_schedule_warm": BETA_HEAVY,
        "n_iters_lse": 10000,
        "n_iters_polyak": 500000,
        "seed": 100,
    }

    print(f"  Launching 2 benchmark containers ({CPU_CORES} cores each)...")
    explore_result = run_explore.remote(explore_config)
    cascade_result = run_cascade.remote(cascade_config)

    # Parse timing
    t_explore = explore_result["elapsed_s"]
    t_cascade = cascade_result["elapsed_s"]

    update_best(global_best, best_per_P,
                explore_result["best_val"], explore_result["best_x"],
                explore_result["P"])
    best_cascade = min(cascade_result["results"], key=lambda r: r[0])
    update_best(global_best, best_per_P,
                best_cascade[0], best_cascade[1],
                cascade_result["P_target"])

    print(f"\n  Explore P=30, 6x10 restarts:  {t_explore:.1f}s  "
          f"best={explore_result['best_val']:.6f}")
    print(f"  Cascade P=100, 1+5 restarts:  {t_cascade:.1f}s  "
          f"best={cascade_result['best_val']:.6f}")

    # Extrapolate full costs
    def scale(bench_time, bench_restarts, full_restarts):
        return bench_time * (full_restarts / bench_restarts)

    # Stage 1: Explore 5 P values, 6 strategies x 15000 restarts each
    # 5 containers in parallel, each running 6 x 15000 = 90,000 restarts
    # Bench ran 6 x 10 = 60 restarts
    explore_per = scale(t_explore, 60, 90000)
    # P=75 ~5x slower than P=30
    # Wall clock = max(P30..P75) since parallel
    explore_wall = explore_per * 5.0  # P=75 is bottleneck
    explore_cost = 5 * explore_per * 2.5 * rate_per_container_sec  # avg 2.5x

    # Stages 2-8: Cascade at P=100, 200, 500, 1000, 1500, 2000, 3000
    # 15 candidates x (1 + 150) = 2265 restarts per level
    # Bench ran 6 restarts at P=100
    cascade_100_per = scale(t_cascade, 6, 151)  # per candidate
    cascade_200_per = cascade_100_per * 2.0
    cascade_500_per = cascade_100_per * 8.0
    cascade_1000_per = cascade_100_per * 25.0
    cascade_1500_per = cascade_100_per * 50.0
    cascade_2000_per = cascade_100_per * 80.0
    cascade_3000_per = cascade_100_per * 160.0

    n_cand_explore = 15  # from exploration
    n_cand_cascade = 5   # after diversity filtering

    c2_cost = n_cand_explore * cascade_100_per * rate_per_container_sec
    c3_cost = n_cand_cascade * cascade_200_per * rate_per_container_sec
    c4_cost = n_cand_cascade * cascade_500_per * rate_per_container_sec
    c5_cost = n_cand_cascade * cascade_1000_per * rate_per_container_sec
    c6_cost = n_cand_cascade * cascade_1500_per * rate_per_container_sec
    c7_cost = n_cand_cascade * cascade_2000_per * rate_per_container_sec
    c8_cost = n_cand_cascade * cascade_3000_per * rate_per_container_sec

    total_cost = (explore_cost + c2_cost + c3_cost + c4_cost
                  + c5_cost + c6_cost + c7_cost + c8_cost)
    bench_cost = max(t_explore, t_cascade) * rate_per_container_sec * 2

    print(f"\n{'='*70}")
    print(f"COST ESTIMATE (based on measured timings)")
    print(f"{'='*70}")
    print(f"  Rate: ~${rate_per_container_sec * 3600:.2f}/container-hr "
          f"({CPU_CORES} cores)")
    print(f"")
    print(f"  Stage 1 (explore P=30-75, 15000x6):     ${explore_cost:6.2f}  "
          f"~{explore_wall/60:.0f} min wall")
    print(f"  Stage 2 (cascade P=100, {n_cand_explore} cands):     ${c2_cost:6.2f}")
    print(f"  Stage 3 (cascade P=200, {n_cand_cascade} cands):      ${c3_cost:6.2f}")
    print(f"  Stage 4 (cascade P=500, {n_cand_cascade} cands):      ${c4_cost:6.2f}")
    print(f"  Stage 5 (cascade P=1000, {n_cand_cascade} cands):     ${c5_cost:6.2f}")
    print(f"  Stage 6 (cascade P=1500, {n_cand_cascade} cands):     ${c6_cost:6.2f}")
    print(f"  Stage 7 (cascade P=2000, {n_cand_cascade} cands):     ${c7_cost:6.2f}")
    print(f"  Stage 8 (cascade P=3000, {n_cand_cascade} cands):     ${c8_cost:6.2f}")
    print(f"  ────────────────────────────────────────────")
    print(f"  TOTAL ESTIMATED COST:                    ${total_cost:6.2f}")
    print(f"  This benchmark cost:                     ~${bench_cost:.2f}")
    print(f"")
    print(f"  Note: P-scaling estimates are approximate.")
    print(f"  Actual cost may vary ~30% due to container startup,")
    print(f"  scheduling overhead, and P-scaling approximations.")
    print(f"")
    print(f"  Best from benchmark: {global_best['val']:.6f}")
    print(f"  Run without --test for the full pipeline.")


# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_full(global_best, best_per_P, explore_diverse, completed_stage):
    """Full curriculum learning pipeline with stage-skipping for --resume."""
    import numpy as np

    EXPLORE_P = [30, 40, 50, 60, 75]
    CASCADE = [100, 200, 500, 1000, 1500, 2000, 3000]
    N_RESTARTS = 15000
    N_WARM = 150
    THRESHOLD = 0.1
    TOP_K_EXPLORE = 15
    TOP_K_CASCADE = 5

    ALL_STAGES = ["explore"] + [f"cascade_{P}" for P in CASCADE]

    def stage_done(name):
        """Check if a stage was already completed (for resume)."""
        if not completed_stage:
            return False
        if completed_stage == "done":
            return True
        try:
            return ALL_STAGES.index(name) <= ALL_STAGES.index(completed_stage)
        except ValueError:
            return False

    # ── Stage 1: Massive exploration at P=30, 40, 50 ──────────────
    if not stage_done("explore"):
        n_strats = len(CURRICULUM_STRATEGIES)
        total = n_strats * N_RESTARTS
        print_header("STAGE 1",
                     f"Massive exploration at P={EXPLORE_P} "
                     f"({n_strats} strategies x {N_RESTARTS:,} = {total:,} restarts each)")

        explore_configs = [
            {
                "P": P,
                "strategies": CURRICULUM_STRATEGIES,
                "n_restarts_per_strategy": N_RESTARTS,
                "beta_schedule": BETA_HEAVY,
                "n_iters_lse": 10000,
                "n_iters_polyak": 200000,
                "diversity_threshold": THRESHOLD,
                "top_k": TOP_K_EXPLORE,
                "seed": 42 + P,
            }
            for P in EXPLORE_P
        ]

        # Launch 3 containers in parallel (one per P)
        results = list(run_explore.map(explore_configs))

        for r in results:
            P = r["P"]
            explore_diverse[str(P)] = r["diverse"]
            update_best(global_best, best_per_P,
                        r["best_val"], r["best_x"], P)
            print(f"  P={P}: best={r['best_val']:.6f} "
                  f"diverse={len(r['diverse'])} time={r['elapsed_s']:.0f}s")

        do_checkpoint(global_best, best_per_P, "explore", explore_diverse)

    # ── Cascade stages ─────────────────────────────────────────────
    # Start from P=50 diverse set
    source_P = max(EXPLORE_P)
    current_diverse = explore_diverse.get(str(source_P), [])
    prev_P = source_P

    if not current_diverse:
        print("ERROR: No diverse solutions from exploration. Cannot cascade.")
        return

    for P_target in CASCADE:
        stage_name = f"cascade_{P_target}"

        if stage_done(stage_name):
            # Skip but recover state for next level
            P_str = str(P_target)
            if P_str in explore_diverse and explore_diverse[P_str]:
                current_diverse = explore_diverse[P_str]
            elif P_str in best_per_P:
                current_diverse = [
                    (best_per_P[P_str]["val"], best_per_P[P_str]["x"])
                ]
            prev_P = P_target
            print(f"  Skipping {stage_name} (already completed)")
            continue

        n_cands = len(current_diverse)
        n_restarts_each = 1 + N_WARM
        print_header(f"CASCADE P={P_target}",
                     f"{n_cands} candidates from P={prev_P}, "
                     f"{n_restarts_each} restarts each "
                     f"({n_cands} containers)")

        # Upsample all candidates locally
        cascade_configs = []
        for cand_idx, (val, x) in enumerate(current_diverse):
            x_up = upsample_local(x, prev_P, P_target)
            cascade_configs.append({
                "P_target": P_target,
                "cand_idx": cand_idx,
                "x_up": x_up,
                "n_warm_restarts": N_WARM,
                "beta_schedule_direct": BETA_ULTRA,
                "beta_schedule_warm": BETA_HEAVY,
                "n_iters_lse": 10000,
                "n_iters_polyak": 500000,
                "seed": 123 + P_target + cand_idx,
            })

        # Launch all candidates in parallel
        cascade_results = list(run_cascade.map(cascade_configs))

        # Collect all solutions
        all_solutions = []
        for cr in cascade_results:
            for val, x in cr["results"]:
                all_solutions.append((val, x))
            print(f"  Candidate #{cr['cand_idx']}: "
                  f"best={cr['best_val']:.6f} time={cr['elapsed_s']:.0f}s")

        # Sort and diversity filter
        all_solutions.sort(key=lambda s: s[0])
        scaled_threshold = THRESHOLD * np.sqrt(50.0 / P_target)
        diverse = keep_diverse_local(
            all_solutions, threshold=scaled_threshold, max_keep=TOP_K_CASCADE
        )

        # Update tracking
        update_best(global_best, best_per_P,
                    all_solutions[0][0], all_solutions[0][1], P_target)

        print(f"\n  Best at P={P_target}: {all_solutions[0][0]:.6f}")
        print(f"  Diverse top-{TOP_K_CASCADE}:")
        for i, (v, _) in enumerate(diverse):
            print(f"    #{i+1}: {v:.6f}")

        # Store and advance
        explore_diverse[str(P_target)] = diverse
        current_diverse = diverse
        prev_P = P_target

        do_checkpoint(global_best, best_per_P, stage_name, explore_diverse)
