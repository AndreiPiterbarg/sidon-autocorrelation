"""Initial GPU bound improvement attempt.

Runs three phases on an A100 GPU:
  A) Warmup & calibrate GPU throughput
  B) Progressive exploration: gpu_find_best_bound_direct(3, m) for increasing m
  C) Formal proof: gpu_run_single_level(3, best_m, c_target)

Results saved incrementally to data/gpu_init_run_{timestamp}.json.
"""
import sys
import os
import time
import json
import math
from datetime import datetime

# Import setup — script runs from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger', 'cpu'))
from gpu.solvers import gpu_find_best_bound_direct, gpu_run_single_level
from gpu.wrapper import is_available, get_device_name
from pruning import correction, count_compositions

# ---------- Configuration ----------
TOTAL_BUDGET_S = 7000       # 2h = 7200s, leave 200s buffer
PHASE_C_RESERVE_S = 600     # 10 min for formal proof
SAFETY_FACTOR = 1.5         # don't start run unless est_time * 1.5 < remaining

# Progressive m values for Phase B (n=3)
M_SCHEDULE = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170, 200, 250, 300]


def log(msg):
    """Timestamped, flushed log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def save_results(results, path):
    """Save results dict to JSON (atomic via temp file)."""
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("data", f"gpu_init_run_{timestamp}.json")
    os.makedirs("data", exist_ok=True)

    results = {
        "timestamp": timestamp,
        "phases": {},
        "best_bound": None,
        "best_m": None,
        "proven": False,
        "proven_bound": None,
    }

    # --- GPU check ---
    log("Checking GPU availability...")
    if not is_available():
        log("ERROR: No CUDA GPU available. Exiting.")
        sys.exit(1)
    dev = get_device_name()
    log(f"GPU: {dev}")
    results["device"] = dev

    def remaining():
        return TOTAL_BUDGET_S - (time.time() - t_start)

    # ================================================================
    #  PHASE A — Warmup & Calibrate
    # ================================================================
    log("=" * 60)
    log("PHASE A: Warmup & Calibrate")
    log("=" * 60)
    phase_a = {}

    # A1: Quick d=4 validation
    log("A1: gpu_find_best_bound_direct(n=2, m=100) — d=4 validation")
    t0 = time.time()
    bound_d4 = gpu_find_best_bound_direct(2, 100, verbose=True)
    elapsed_d4 = time.time() - t0
    log(f"A1 done: bound={bound_d4:.6f}, time={elapsed_d4:.1f}s")
    phase_a["d4_validation"] = {
        "n": 2, "m": 100, "bound": bound_d4,
        "elapsed": elapsed_d4,
        "grid_points": count_compositions(4, 200),
    }

    # A2: Calibrate d=6 throughput
    log("A2: gpu_find_best_bound_direct(n=3, m=30) — d=6 calibration")
    t0 = time.time()
    bound_cal = gpu_find_best_bound_direct(3, 30, verbose=True)
    elapsed_cal = time.time() - t0
    grid_cal = count_compositions(6, 90)
    rate_cal = grid_cal / elapsed_cal if elapsed_cal > 0 else 0
    log(f"A2 done: bound={bound_cal:.6f}, time={elapsed_cal:.1f}s, "
        f"rate={rate_cal:.0f} configs/s")
    phase_a["d6_calibration"] = {
        "n": 3, "m": 30, "bound": bound_cal,
        "elapsed": elapsed_cal,
        "grid_points": grid_cal,
        "rate": rate_cal,
    }

    results["phases"]["A"] = phase_a
    save_results(results, out_path)

    # ================================================================
    #  PHASE B — Progressive Exploration
    # ================================================================
    log("=" * 60)
    log("PHASE B: Progressive Exploration (n=3)")
    log("=" * 60)
    phase_b = []
    best_bound = bound_cal
    best_m = 30

    # Exponential moving average for rate estimation
    ema_rate = rate_cal
    ema_alpha = 0.5

    for m in M_SCHEDULE:
        grid = count_compositions(6, 12 * m)  # d=6, S=4*3*m=12m
        est_time = grid / ema_rate if ema_rate > 0 else float('inf')
        time_left = remaining() - PHASE_C_RESERVE_S

        log(f"--- m={m}: grid={grid:,}, est_time={est_time:.0f}s, "
            f"time_left={time_left:.0f}s ---")

        if est_time * SAFETY_FACTOR > time_left:
            log(f"Skipping m={m}: estimated {est_time * SAFETY_FACTOR:.0f}s > "
                f"{time_left:.0f}s remaining")
            break

        t0 = time.time()
        bound = gpu_find_best_bound_direct(3, m, verbose=True)
        elapsed = time.time() - t0
        actual_rate = grid / elapsed if elapsed > 0 else ema_rate

        # Update EMA rate
        ema_rate = ema_alpha * actual_rate + (1 - ema_alpha) * ema_rate

        corr = correction(m)
        entry = {
            "m": m, "bound": bound, "elapsed": elapsed,
            "grid_points": grid, "rate": actual_rate,
            "correction": corr,
        }
        phase_b.append(entry)
        log(f"m={m}: bound={bound:.6f}, time={elapsed:.1f}s, "
            f"rate={actual_rate:.0f} cfg/s")

        if bound > best_bound:
            best_bound = bound
            best_m = m
            log(f"  *** New best: C_{{1a}} >= {best_bound:.6f} at m={best_m}")

        results["phases"]["B"] = phase_b
        results["best_bound"] = best_bound
        results["best_m"] = best_m
        save_results(results, out_path)

    log(f"Phase B complete. Best bound: {best_bound:.6f} at m={best_m}")

    # ================================================================
    #  PHASE C — Formal Proof
    # ================================================================
    log("=" * 60)
    log("PHASE C: Formal Proof")
    log("=" * 60)
    phase_c = {}

    # Round down to 4 decimal places for a clean target
    c_target = math.floor(best_bound * 10000) / 10000.0
    log(f"Best bound from exploration: {best_bound:.6f}")
    log(f"Initial proof target: c_target = {c_target:.4f}")

    # Try proving, backing off if it fails
    max_attempts = 5
    for attempt in range(max_attempts):
        if remaining() < 60:
            log("Not enough time for proof attempt. Skipping.")
            break

        log(f"Proof attempt {attempt + 1}: c_target = {c_target:.4f} with m={best_m}")
        t0 = time.time()
        result = gpu_run_single_level(3, best_m, c_target, verbose=True)
        elapsed = time.time() - t0

        attempt_info = {
            "attempt": attempt + 1,
            "c_target": c_target,
            "m": best_m,
            "proven": result["proven"],
            "n_survivors": result["n_survivors"],
            "elapsed": elapsed,
        }
        phase_c[f"attempt_{attempt + 1}"] = attempt_info

        if result["proven"]:
            log(f"PROVEN: C_{{1a}} >= {c_target:.4f}")
            results["proven"] = True
            results["proven_bound"] = c_target
            results["phases"]["C"] = phase_c
            save_results(results, out_path)
            break
        else:
            log(f"NOT proven at {c_target:.4f} "
                f"({result['n_survivors']} survivors). Reducing target.")
            c_target -= 0.001
            if c_target < 1.0:
                log("Target dropped below 1.0, stopping.")
                break

    results["phases"]["C"] = phase_c
    save_results(results, out_path)

    # ================================================================
    #  Final Summary
    # ================================================================
    total_elapsed = time.time() - t_start
    log("=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    log(f"Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    log(f"Device: {dev}")
    log(f"Best exploratory bound: C_{{1a}} >= {best_bound:.6f} (n=3, m={best_m})")
    if results["proven"]:
        log(f"FORMALLY PROVEN: C_{{1a}} >= {results['proven_bound']:.4f}")
    else:
        log("No formal proof achieved.")
    log(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
