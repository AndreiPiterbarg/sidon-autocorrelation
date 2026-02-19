"""Hierarchical branch-and-prune proof on RunPod A100 GPU.

Implements the full Cloninger-Steinerberger algorithm (arXiv:1403.7988):

  1. Start at coarse level n=n_start, enumerate all b in B_{n,m}
  2. Prune via asymmetry + canonical filtering
  3. Test all windows — rule out parents whose max windowed autoconvolution > T
     where T = c_target + 2/m + 1/m^2
  4. Extract survivors (parents NOT ruled out)
  5. Refine survivors: each parent's bins are split (n -> 2n), generating all
     child vectors. Parent is ruled out iff ALL children are ruled out.
  6. Repeat until all eliminated or budget exhausted

This replaces init_test_run.py, which only did single-level work (no refinement).

Results saved incrementally to data/gpu_proof_{timestamp}.json.

Usage:
    python run_proof.py                           # defaults: c=1.10, m=50, n=3
    python run_proof.py --target 1.20 --m 80      # custom target and resolution
    python run_proof.py --resume                   # resume from last checkpoint
"""
import sys
import os
import time
import json
import argparse
import platform
from datetime import datetime

# Import setup — script runs from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger', 'cpu'))
from gpu.solvers import gpu_find_best_bound_direct, gpu_run_single_level
from gpu.wrapper import (is_available, get_device_name, refine_parents,
                         max_survivors_for_dim, get_free_memory,
                         load_survivors_chunk)
from pruning import correction, count_compositions, asymmetry_threshold


# ---------- Configuration ----------
TOTAL_BUDGET_S = 54000      # 15h for multi-level runs
WARMUP_RESERVE_S = 120      # time for warmup phase
CHECKPOINT_DIR = 'data'


def log(msg):
    """Timestamped, flushed log line."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def save_results(results, path):
    """Save results dict to JSON (atomic via temp file).

    Also prints a machine-readable checkpoint line to stdout so that
    the local log file (captured by gpupod) preserves structured data
    even if the SSH connection drops before teardown.
    """
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, path)
    # Emit checkpoint to stdout for local log capture
    print(f"===CHECKPOINT_JSON==={json.dumps(results, default=str)}===END_CHECKPOINT===",
          flush=True)


def fmt_count(n):
    """Format large counts with commas."""
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(
        description='Hierarchical branch-and-prune proof (CS14 algorithm)')
    parser.add_argument('--target', type=float, default=1.10,
                        help='Target lower bound c_target (default: 1.10)')
    parser.add_argument('--m', type=int, default=50,
                        help='Grid resolution m (default: 50, paper used 50)')
    parser.add_argument('--n-start', type=int, default=3,
                        help='Starting n (d=2n bins, default: 3)')
    parser.add_argument('--max-levels', type=int, default=4,
                        help='Max refinement levels (default: 4, giving n=3,6,12,24)')
    parser.add_argument('--time-budget', type=float, default=TOTAL_BUDGET_S,
                        help=f'Time budget in seconds (default: {TOTAL_BUDGET_S})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR,
                        help=f'Checkpoint directory (default: {CHECKPOINT_DIR})')
    parser.add_argument('--force', action='store_true',
                        help='Run even if estimated time exceeds budget')
    args = parser.parse_args()

    c_target = args.target
    m = args.m
    n_start = args.n_start
    max_levels = args.max_levels
    time_budget = args.time_budget
    checkpoint_dir = args.checkpoint_dir

    t_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(checkpoint_dir, f"gpu_proof_{timestamp}.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ================================================================
    #  Algorithm parameters (Section 3 of the math doc)
    # ================================================================
    corr = correction(m)                         # 2/m + 1/m^2
    T = c_target + corr                          # effective threshold
    asym_thresh = asymmetry_threshold(c_target)   # sqrt(c_target / 2)
    level_schedule = [n_start * (2 ** lvl) for lvl in range(max_levels)]

    results = {
        "timestamp": timestamp,
        "algorithm": "Cloninger-Steinerberger hierarchical branch-and-prune",
        "strict_fail_closed": True,
        "status": "inconclusive",
        "inconclusive_reason": "run_in_progress",
        "parameters": {
            "c_target": c_target,
            "m": m,
            "n_start": n_start,
            "max_levels": max_levels,
            "correction": corr,
            "effective_threshold_T": T,
            "asymmetry_threshold": asym_thresh,
            "level_schedule_n": level_schedule,
            "level_schedule_d": [2 * n for n in level_schedule],
        },
        "levels": [],
        "proven": False,
        "proven_bound": None,
    }

    log("=" * 64)
    log("HIERARCHICAL BRANCH-AND-PRUNE PROOF")
    log("  Algorithm: Cloninger-Steinerberger (arXiv:1403.7988)")
    log("=" * 64)
    log(f"  c_target   = {c_target}")
    log(f"  m          = {m}")
    log(f"  correction = 2/m + 1/m^2 = {corr:.6f}")
    log(f"  threshold T = c_target + correction = {T:.6f}")
    log(f"  asymmetry  = sqrt(c_target/2) = {asym_thresh:.6f}")
    log(f"  levels     = n in {level_schedule} (d in {[2*n for n in level_schedule]})")
    log(f"  time budget = {time_budget:.0f}s")
    log(f"  mode       = strict fail-closed")
    log("")

    # --- GPU check ---
    log("Checking GPU availability...")
    if not is_available():
        log("ERROR: No CUDA GPU available. Exiting.")
        sys.exit(1)
    dev = get_device_name()
    log(f"GPU: {dev}")
    results["device"] = dev

    def remaining():
        return time_budget - (time.time() - t_start)

    def _set_status(status, reason=None):
        results["status"] = status
        results["inconclusive_reason"] = reason

    def _fail_closed(reason, n_remaining=None):
        log(f"  FATAL INCONCLUSIVE [{reason}]")
        _set_status("inconclusive", reason)
        results["proven"] = False
        results["proven_bound"] = None
        if n_remaining is not None:
            results["remaining_survivors"] = n_remaining
        try:
            save_results(results, out_path)
        except OSError as e:
            log(f"  WARNING: Results save failed during fail-closed: {e}")
        _print_summary(results, time.time() - t_start, dev)
        sys.exit(2)

    # ================================================================
    #  PHASE A — Warmup (quick validation run)
    # ================================================================
    log("")
    log("=" * 64)
    log("PHASE A: GPU Warmup")
    log("=" * 64)

    # Detect platform — Windows has TDR (kills GPU kernels after ~2s)
    is_windows = platform.system() == 'Windows'
    if is_windows:
        log("Platform: Windows (TDR watchdog active — kernel time limit ~2s)")
    else:
        log("Platform: Linux (no TDR — long kernels OK)")

    # Calibrate GPU throughput with a small d=6 run (same dimension as level 0)
    # Use small m to avoid TDR on Windows
    cal_m = 10 if is_windows else 30
    cal_n = n_start
    cal_d = 2 * cal_n
    cal_S = cal_m  # S=m convention
    cal_grid = count_compositions(cal_d, cal_S)
    log(f"Calibrating: gpu_find_best_bound_direct(n={cal_n}, m={cal_m}) — "
        f"{fmt_count(cal_grid)} points")
    t0 = time.time()
    bound_warmup = gpu_find_best_bound_direct(cal_n, cal_m, verbose=False)
    elapsed_warmup = time.time() - t0
    cal_rate = cal_grid / elapsed_warmup if elapsed_warmup > 0 else 1e9
    log(f"Calibration done: bound={bound_warmup:.6f}, time={elapsed_warmup:.1f}s, "
        f"rate={cal_rate:.0f} configs/s")
    results["warmup"] = {
        "bound": bound_warmup,
        "elapsed": elapsed_warmup,
        "calibration_rate": cal_rate,
        "platform": platform.system(),
    }
    save_results(results, out_path)

    # Pre-flight check: estimate level 0 time and abort if infeasible
    level0_S = m  # S=m convention
    level0_grid = count_compositions(2 * n_start, level0_S)
    est_level0_time = level0_grid / cal_rate

    # On Windows, the CUDA kernel runs as ONE uninterruptible launch.
    # Windows TDR kills any GPU kernel running > ~2 seconds on a display GPU.
    # On Linux (RunPod), there's no TDR — only the time budget matters.
    if is_windows:
        max_kernel_time = 2.0  # TDR default
    else:
        max_kernel_time = remaining() * 0.9

    log("")
    log(f"Pre-flight estimate for Level 0 (n={n_start}, m={m}):")
    log(f"  Grid size: {fmt_count(level0_grid)}")
    log(f"  Estimated kernel time: {est_level0_time:.1f}s ({est_level0_time / 60:.1f} min)")
    if is_windows:
        log(f"  Windows TDR limit: ~{max_kernel_time:.0f}s "
            f"(kernel WILL crash if it exceeds this)")
    else:
        log(f"  Time budget: {remaining():.0f}s ({remaining() / 60:.1f} min)")

    if est_level0_time > max_kernel_time:
        # Find the largest m that fits within the time limit
        max_feasible_m = 10
        for try_m in range(m - 1, 9, -1):
            try_grid = count_compositions(2 * n_start, try_m)  # S=m
            if try_grid / cal_rate < max_kernel_time * 0.8:
                max_feasible_m = try_m
                break

        if is_windows:
            log(f"  FATAL: m={m} will cause a TDR crash on Windows!")
            log(f"  The GPU kernel takes ~{est_level0_time:.0f}s but Windows kills "
                f"it after ~2s.")
            log(f"  Max feasible m for this GPU: ~{max_feasible_m}")
            log(f"  For large m, use RunPod (Linux, no TDR).")
        else:
            log(f"  WARNING: Level 0 will likely exceed the time budget!")
            log(f"  The base-level CUDA kernel runs in one uninterruptible launch.")
            log(f"  Suggested max feasible m ~ {max_feasible_m} for this budget.")

        if not args.force:
            log(f"  ABORTING. Use --force to run anyway, or reduce --m.")
            log(f"  Example: python run_proof.py --target {c_target} --m {max_feasible_m}")
            save_results(results, out_path)
            sys.exit(1)
        else:
            log(f"  --force set, proceeding anyway.")

    # ================================================================
    #  PHASE B — Hierarchical Branch-and-Prune (the actual algorithm)
    #
    #  Section 11 pseudocode from mathematical_algo_explanation.md:
    #
    #    survivors = enumerate_all(B_{n_start, m})
    #    apply asymmetry pruning + canonical filtering
    #    n = n_start
    #    while survivors not empty and n <= n_max:
    #        for each parent b in survivors:
    #            generate all N = prod(b_i + 1) refinements  (S=m convention)
    #            test each refinement against threshold T
    #            parent ruled out iff ALL refinements ruled out
    #        survivors = surviving parents' children
    #        n = 2*n
    #    if survivors empty: PROVEN
    # ================================================================
    log("")
    log("=" * 64)
    log("PHASE B: Hierarchical Proof")
    log("=" * 64)

    survivors = None
    survivor_file = None  # path to binary file when using streamed mode
    start_level = 0
    n_survived = 0  # track for end summary when survivors are on disk

    # Resume from checkpoint if requested
    if args.resume:
        start_level, survivors, survivor_file = _try_resume(checkpoint_dir, c_target, m, n_start)
        if survivor_file is not None:
            log(f"Resumed from level {start_level - 1} checkpoint "
                f"(survivors on disk: {survivor_file})")
        elif survivors is not None:
            log(f"Resumed from level {start_level - 1} checkpoint "
                f"({fmt_count(len(survivors))} survivors)")
        else:
            log("No compatible checkpoint found to resume.")

    for level in range(start_level, max_levels):
        n_half = n_start * (2 ** level)
        d = 2 * n_half

        time_left = remaining()
        if time_left <= 30:
            _fail_closed(f"time_budget_exhausted_before_level_{level}")

        log("")
        log(f"--- Level {level}: n={n_half}, d={d} bins ---")
        log(f"  Time remaining: {time_left:.0f}s")
        t_level = time.time()

        if level == 0:
            # ============================================================
            # Level 0: Base enumeration — single-pass extraction
            #
            # For large runs (D>=6, m>=20), uses chunked streaming to
            # disk to avoid OOM. For small runs, uses in-memory extraction.
            # ============================================================
            import numpy as np
            S = m  # S=m convention
            n_total = count_compositions(d, S)
            log(f"  Grid B_{{n,m}}: {fmt_count(n_total)} lattice points")
            log(f"  S = m = {S}")
            log(f"  Prune threshold T = {T:.6f}")
            log(f"  Asymmetry threshold = {asym_thresh:.6f}")

            free_gb = get_free_memory() / (1024 ** 3)
            log(f"  GPU free memory: {free_gb:.1f} GB")

            result = gpu_run_single_level(
                n_half, m, c_target,
                verbose=True,
                extract_survivors=True)

            use_streamed = result['stats'].get('streamed', False)
            n_ext = result['stats'].get('n_extracted', 0)
            survivor_file = result.get('survivor_file', None)

            if result['n_survivors'] > 0 and n_ext < result['n_survivors']:
                _fail_closed("base_extraction_truncated",
                             n_remaining=result['n_survivors'])
            if use_streamed and result['n_survivors'] > 0 and not survivor_file:
                _fail_closed("base_streamed_missing_survivor_file",
                             n_remaining=result['n_survivors'])

            level_elapsed = time.time() - t_level
            n_survived = result['n_survivors']

            level_info = {
                "level": level,
                "n_half": n_half,
                "d": d,
                "type": "base_enumeration",
                "grid_points": n_total,
                "n_pruned_asym": result['stats']['n_pruned_asym'],
                "n_pruned_test": result['stats']['n_pruned_test'],
                "n_survivors": n_survived,
                "n_extracted": n_ext,
                "streamed": use_streamed,
                "survivor_file": survivor_file,
                "elapsed": level_elapsed,
            }

            log(f"  Level 0 results:")
            log(f"    FP32 pre-checks skipped: {fmt_count(result['stats'].get('n_fp32_skipped', 0))}")
            log(f"    Asymmetry pruned (FP64 final): {fmt_count(result['stats']['n_pruned_asym'])}")
            log(f"    Test-value pruned: {fmt_count(result['stats']['n_pruned_test'])}")
            log(f"    Survivors: {fmt_count(n_survived)}")
            if use_streamed and survivor_file:
                log(f"    Streamed to: {survivor_file}")
            log(f"    Elapsed: {level_elapsed:.1f}s")

            if result['proven']:
                log("")
                log(f"  >>> PROVEN at base level: c >= {c_target} <<<")
                results["levels"].append(level_info)
                results["proven"] = True
                results["proven_bound"] = c_target
                _set_status("proven", None)
                try:
                    save_results(results, out_path)
                except OSError as e:
                    log(f"  WARNING: Results save failed: {e}")
                _print_summary(results, time.time() - t_start, dev)
                return

            # For streamed mode, survivors live on disk
            if use_streamed and survivor_file:
                survivors = None  # marker: use survivor_file instead
            else:
                survivors = result['survivors']
                survivor_file = None

        else:
            # ============================================================
            # Level 1+: Refinement (Section 4, Steps 3-4)
            #
            # For each surviving parent b from the previous level:
            #   - Each parent component b_i is split into two sub-components
            #     c_{2i} and c_{2i+1} with c_{2i} + c_{2i+1} = b_i  (S=m convention)
            #   - Number of ways to split: (b_i + 1) per component
            #   - Total refinements: N = prod_i(b_i + 1)
            #
            # A parent is ruled out iff EVERY one of its N refinements
            # is ruled out (i.e., every refinement has at least one
            # window exceeding T).
            #
            # When survivors are on disk (streamed mode), process them
            # in batches to bound host memory.
            # ============================================================
            import numpy as np

            # Determine whether survivors are in-memory or on disk
            use_file_mode = (survivors is None and survivor_file is not None)

            if use_file_mode:
                import os as _os
                file_size = _os.path.getsize(survivor_file)
                d_parent = d // 2
                per_surv = d_parent * 4  # bytes per survivor (parent dim)
                num_parents = file_size // per_surv
            else:
                if survivors is None or len(survivors) == 0:
                    _fail_closed("empty_survivor_set_without_completion")
                d_parent = d // 2
                num_parents = len(survivors)

            log(f"  Parents from level {level - 1}: {fmt_count(num_parents)}")
            log(f"  Parent dimension: d_parent={d_parent}")
            log(f"  Child dimension:  d_child={d}")
            if use_file_mode:
                log(f"  Mode: CHUNKED FILE REFINEMENT (survivors on disk)")
                log(f"  Survivor file: {survivor_file}")

            # ============================================================
            # Pre-flight estimation: sample survivors, estimate total work
            # ============================================================
            PREFLIGHT_SAMPLE = 10000
            if use_file_mode:
                sample_size = min(PREFLIGHT_SAMPLE, num_parents)
                sample = load_survivors_chunk(survivor_file, d_parent, 0, sample_size)
            else:
                sample_size = min(PREFLIGHT_SAMPLE, num_parents)
                sample = survivors[:sample_size]

            # With S=m + energy cap: effective split count per component.
            # x_cap = floor(m * sqrt(thresh / d_child)); any sub-bin > x_cap
            # is pruned by ell=2 check, so we skip generating those children.
            corr_ref = 2.0 / args.m + 1.0 / (args.m * args.m)
            thresh_ref = c_target + corr_ref + 1e-9  # prune_target + fp_margin
            x_cap = int(np.floor(args.m * np.sqrt(thresh_ref / d)))
            x_cap = min(x_cap, args.m)
            # Effective splits per component: [max(0,B[i]-x_cap), min(B[i],x_cap)]
            sample_f = sample.astype(np.float64)
            lo = np.maximum(0.0, sample_f - x_cap)
            hi = np.minimum(sample_f, float(x_cap))
            eff_counts = hi - lo + 1.0
            eff_counts = np.maximum(eff_counts, 0.0)  # guard negative
            refs_per_sample = np.prod(eff_counts, axis=1)
            avg_refs = float(refs_per_sample.mean())
            max_refs = float(refs_per_sample.max())
            total_refs_est = avg_refs * num_parents
            log(f"  Energy cap x_cap={x_cap} (m={args.m}, thresh={thresh_ref:.6f}, d={d})")

            log(f"  Pre-flight refinement estimate (sampled {fmt_count(sample_size)} parents):")
            log(f"    Avg refinements/parent: {avg_refs:.2e}")
            log(f"    Max refinements/parent: {max_refs:.2e}")
            log(f"    Total refinements est:  {total_refs_est:.2e}")

            # Calibrate refinement throughput by running a small sample.
            # The base-level cal_rate is NOT representative — refinement is
            # ~1000x faster because most children are eliminated by cheap
            # FP32 pre-checks before reaching the expensive autoconvolution.
            CAL_REFINE_PARENTS = min(500, num_parents)
            # Use evenly-spaced sample for representativeness
            cal_indices = np.linspace(0, num_parents - 1,
                                      CAL_REFINE_PARENTS, dtype=int)
            if use_file_mode:
                cal_parents = np.empty((CAL_REFINE_PARENTS, d_parent),
                                       dtype=np.int32)
                for ci, idx in enumerate(cal_indices):
                    chunk = load_survivors_chunk(
                        survivor_file, d_parent, int(idx), 1)
                    cal_parents[ci] = chunk[0]
            else:
                cal_parents = survivors[cal_indices]

            log(f"  Calibrating refinement rate ({CAL_REFINE_PARENTS} parents)...")
            t_cal = time.time()
            cal_result = refine_parents(
                d_parent=d_parent,
                parent_configs_array=cal_parents,
                m=m,
                c_target=c_target,
                max_survivors=10000,
                time_budget_sec=30)
            cal_elapsed = time.time() - t_cal
            cal_total_refs = (cal_result['total_asym'] +
                              cal_result['total_test'] +
                              cal_result['total_survivors'])

            if cal_elapsed > 0.01 and cal_total_refs > 0:
                ref_rate = cal_total_refs / cal_elapsed
                est_refine_secs = total_refs_est / ref_rate
                log(f"    Calibration: {cal_total_refs:.2e} refs in {cal_elapsed:.2f}s "
                    f"= {ref_rate:.2e} refs/s")
            else:
                # Fallback: use base rate with a conservative multiplier
                ref_rate = cal_rate * 100
                est_refine_secs = total_refs_est / ref_rate
                log(f"    Calibration too fast to measure, using base rate * 100")

            log(f"    Estimated time:         {est_refine_secs:.0f}s "
                f"({est_refine_secs / 3600:.2f} hrs)")

            if not use_file_mode:
                total_refs = total_refs_est
                log(f"  Total refinements to check: ~{total_refs:.2e}")

            if est_refine_secs > remaining() and not args.force:
                log(f"  INFEASIBLE: estimated {est_refine_secs/3600:.1f} hrs "
                    f"but only {remaining()/3600:.1f} hrs budget remaining")
                log(f"  Options:")
                log(f"    1. Use --force to attempt anyway (will time out)")
                log(f"    2. Lower --target (fewer survivors = less work)")
                log(f"    3. Lower --m (fewer refinements per parent)")
                log(f"  ABORTING refinement. Level 0 results are saved.")
                level_info = {
                    "level": level, "n_half": n_half, "d": d,
                    "type": "refinement_skipped",
                    "reason": "infeasible",
                    "est_total_refs": total_refs_est,
                    "est_time_secs": est_refine_secs,
                    "num_parents": num_parents,
                    "avg_refs_per_parent": avg_refs,
                    "elapsed": time.time() - t_level,
                }
                results["levels"].append(level_info)
                results["remaining_survivors"] = num_parents
                try:
                    save_results(results, out_path)
                except OSError as e:
                    log(f"  WARNING: Results save failed: {e}")
                _fail_closed("refinement_aborted_infeasible", n_remaining=num_parents)
            elif est_refine_secs > remaining():
                log(f"  WARNING: estimated {est_refine_secs/3600:.1f} hrs exceeds "
                    f"budget {remaining()/3600:.1f} hrs — proceeding with --force")

            log(f"  Each parent ruled out iff ALL its refinements ruled out")

            ref_buf_size = max_survivors_for_dim(d)
            ref_buf_gb = ref_buf_size * d * 4 / (1024 ** 3)
            log(f"  Refinement buffer: {fmt_count(ref_buf_size)} ({ref_buf_gb:.1f} GB)")

            if use_file_mode:
                # Chunked refinement: read batches from disk
                REFINE_BATCH = 20_000_000  # 20M parents * 24 bytes = 480 MB
                total_asym = 0
                total_test = 0
                total_survived = 0
                total_ref_extracted = 0
                min_tv = float('inf')
                min_cfg = None
                all_child_survivors = []
                timed_out = False
                batch_idx = 0

                log(f"  Batch size: {fmt_count(REFINE_BATCH)} parents")
                log(f"  Estimated batches: {(num_parents + REFINE_BATCH - 1) // REFINE_BATCH}")

                for start in range(0, num_parents, REFINE_BATCH):
                    end = min(start + REFINE_BATCH, num_parents)
                    chunk = load_survivors_chunk(
                        survivor_file, d_parent, start, end - start)

                    batch_time_budget = max(time_left - 30, 60)
                    batch_result = refine_parents(
                        d_parent=d_parent,
                        parent_configs_array=chunk,
                        m=m,
                        c_target=c_target,
                        max_survivors=ref_buf_size,
                        time_budget_sec=batch_time_budget)

                    batch_total_survivors = batch_result['total_survivors']
                    batch_n_extracted = batch_result['n_extracted']
                    batch_overflow = batch_total_survivors > batch_n_extracted
                    batch_exact = batch_total_survivors == batch_n_extracted
                    batch_inconsistent = batch_n_extracted > batch_total_survivors

                    total_asym += batch_result['total_asym']
                    total_test += batch_result['total_test']
                    total_survived += batch_total_survivors
                    total_ref_extracted += batch_n_extracted

                    if batch_result['min_test_val'] < min_tv:
                        min_tv = batch_result['min_test_val']
                        min_cfg = batch_result['min_test_config']

                    if batch_result['n_extracted'] > 0:
                        all_child_survivors.append(
                            batch_result['survivor_configs'])

                    log(f"    Batch {batch_idx}: parents [{fmt_count(start)}, "
                        f"{fmt_count(end)}), surv={fmt_count(batch_result['total_survivors'])}, "
                        f"cumulative={fmt_count(total_survived)}")

                    if batch_result['timed_out']:
                        timed_out = True
                        _fail_closed(
                            f"refinement_batch_timed_out_level_{level}_batch_{batch_idx}",
                            n_remaining=total_survived)
                    if batch_overflow:
                        _fail_closed(
                            f"refinement_batch_extraction_truncated_level_{level}_batch_{batch_idx}",
                            n_remaining=total_survived)
                    if batch_inconsistent:
                        _fail_closed(
                            f"refinement_batch_extraction_count_inconsistent_level_{level}_batch_{batch_idx}",
                            n_remaining=total_survived)
                    # batch_exact means no overflow and is safe to continue.
                    if batch_exact:
                        pass

                    time_left = remaining()
                    if time_left <= 30:
                        timed_out = True
                        _fail_closed(
                            f"time_budget_exhausted_during_refinement_level_{level}",
                            n_remaining=total_survived)

                    batch_idx += 1

                # Combine child survivors from all batches
                if all_child_survivors:
                    child_survivors = np.concatenate(all_child_survivors, axis=0)
                else:
                    child_survivors = np.empty((0, d), dtype=np.int32)

                level_elapsed = time.time() - t_level
                n_survived = total_survived

                level_info = {
                    "level": level,
                    "n_half": n_half,
                    "d": d,
                    "type": "refinement_chunked",
                    "num_parents": num_parents,
                    "n_pruned_asym": total_asym,
                    "n_pruned_test": total_test,
                    "n_survivors": n_survived,
                    "n_extracted": total_ref_extracted,
                    "timed_out": timed_out,
                    "batches": batch_idx + 1,
                    "elapsed": level_elapsed,
                }

            else:
                # In-memory refinement (original path)
                ref_result = refine_parents(
                    d_parent=d_parent,
                    parent_configs_array=survivors,
                    m=m,
                    c_target=c_target,
                    max_survivors=ref_buf_size,
                    time_budget_sec=max(time_left - 30, 60))

                level_elapsed = time.time() - t_level
                n_survived = ref_result['total_survivors']
                n_extracted = ref_result['n_extracted']
                timed_out = ref_result['timed_out']
                extraction_overflow = n_survived > n_extracted
                extraction_exact = n_survived == n_extracted
                extraction_inconsistent = n_extracted > n_survived
                child_survivors = ref_result['survivor_configs']
                if timed_out:
                    _fail_closed(
                        f"refinement_timed_out_level_{level}",
                        n_remaining=n_survived)
                if extraction_overflow:
                    _fail_closed(
                        f"refinement_extraction_truncated_level_{level}",
                        n_remaining=n_survived)
                if extraction_inconsistent:
                    _fail_closed(
                        f"refinement_extraction_count_inconsistent_level_{level}",
                        n_remaining=n_survived)
                # extraction_exact means no overflow and is safe to continue.
                if extraction_exact:
                    pass

                level_info = {
                    "level": level,
                    "n_half": n_half,
                    "d": d,
                    "type": "refinement",
                    "num_parents": num_parents,
                    "total_refinements": total_refs,
                    "n_pruned_asym": ref_result['total_asym'],
                    "n_pruned_test": ref_result['total_test'],
                    "n_survivors": n_survived,
                    "n_extracted": n_extracted,
                    "timed_out": timed_out,
                    "extraction_truncated": extraction_overflow,
                    "elapsed": level_elapsed,
                }

            log(f"  Level {level} results:")
            log(f"    Asymmetry pruned: {fmt_count(level_info['n_pruned_asym'])}")
            log(f"    Test-value pruned: {fmt_count(level_info['n_pruned_test'])}")
            log(f"    Survivors: {fmt_count(n_survived)}")
            if level_info['n_extracted'] > 0:
                log(f"    Extracted: {fmt_count(level_info['n_extracted'])} configs")
            log(f"    Elapsed: {level_elapsed:.1f}s")

            if timed_out:
                _fail_closed(f"refinement_timed_out_level_{level}",
                             n_remaining=n_survived)

            if n_survived == 0:
                log("")
                log(f"  >>> PROVEN at level {level}: c >= {c_target} <<<")
                results["levels"].append(level_info)
                results["proven"] = True
                results["proven_bound"] = c_target
                _set_status("proven", None)
                try:
                    save_results(results, out_path)
                except OSError as e:
                    log(f"  WARNING: Results save failed: {e}")
                _print_summary(results, time.time() - t_start, dev)
                return

            survivors = child_survivors
            survivor_file = None  # clear file mode for next level

        results["levels"].append(level_info)

        # Emit critical results to stdout immediately (survives disk failures)
        log(f"  ===LEVEL_RESULT=== level={level} survivors={n_survived} "
            f"elapsed={level_elapsed:.1f}s")

        # Save results JSON FIRST (tiny, most important)
        try:
            save_results(results, out_path)
        except OSError as e:
            log(f"  WARNING: Results save failed: {e}")

        # Then checkpoint (large, less critical)
        if survivors is not None and len(survivors) > 0:
            checkpoint_bytes = len(survivors) * d * 4  # int32 per element
            CHECKPOINT_MAX_BYTES = 1 * 1024 ** 3  # 1 GB
            if checkpoint_bytes <= CHECKPOINT_MAX_BYTES:
                try:
                    _save_checkpoint(checkpoint_dir, level, survivors, {
                        "c_target": c_target, "m": m, "n_start": n_start,
                        "level": level, "n_half": n_half, "d": d,
                        "n_survivors": len(survivors),
                        "storage_mode": "memory",
                    })
                    log(f"  Checkpoint saved: {fmt_count(len(survivors))} survivors")
                except OSError as e:
                    log(f"  WARNING: Checkpoint save failed (disk full?): {e}")
            else:
                log(f"  Checkpoint skipped: {checkpoint_bytes / (1024**3):.1f} GB "
                    f"exceeds 1 GB disk limit (survivors on disk)")
        elif survivor_file is not None:
            log(f"  Survivors on disk: {survivor_file}")
            try:
                _save_checkpoint(checkpoint_dir, level, np.empty((0,), dtype=np.int32), {
                    "c_target": c_target, "m": m, "n_start": n_start,
                    "level": level, "n_half": n_half, "d": d,
                    "n_survivors": n_survived,
                    "storage_mode": "file",
                    "survivor_file_path": survivor_file,
                })
            except OSError as e:
                log(f"  WARNING: Checkpoint metadata save failed: {e}")

    # ================================================================
    #  If we get here, not all survivors were eliminated
    # ================================================================
    total_elapsed = time.time() - t_start
    n_remaining = len(survivors) if survivors is not None else n_survived
    results["remaining_survivors"] = n_remaining
    _set_status("not_proven", None)
    try:
        save_results(results, out_path)
    except OSError as e:
        log(f"  WARNING: Final results save failed: {e}")
    _print_summary(results, total_elapsed, dev)


def _try_resume(checkpoint_dir, c_target, m, n_start):
    """Try to resume from a previous checkpoint.

    Returns (start_level, survivors, survivor_file_path). Exactly one of
    survivors or survivor_file_path will be non-None on success.
    """
    import numpy as np

    for lvl in range(10, -1, -1):
        meta_path = os.path.join(checkpoint_dir, f'level_{lvl}_meta.json')
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if (meta.get('c_target') != c_target or
                meta.get('m') != m or
                meta.get('n_start') != n_start):
            continue

        storage_mode = meta.get('storage_mode', 'memory')
        streamed_path = meta.get('survivor_file_path')
        if storage_mode == "file" or (streamed_path is not None):
            if streamed_path and os.path.exists(streamed_path):
                return lvl + 1, None, streamed_path
            # Stale streamed checkpoint; try older levels.
            continue

        surv_path = meta.get('checkpoint_survivor_npy')
        if not surv_path:
            surv_path = os.path.join(checkpoint_dir, f'level_{lvl}_survivors.npy')
        if not os.path.exists(surv_path):
            continue
        survivors = np.load(surv_path)
        if survivors.size == 0 and meta.get('n_survivors', 0) > 0:
            # Inconsistent checkpoint; try older levels.
            continue
        return lvl + 1, survivors, None
    return 0, None, None


def _save_checkpoint(checkpoint_dir, level, survivors, metadata):
    """Save survivors and metadata for a completed level."""
    import numpy as np

    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_path = os.path.join(checkpoint_dir, f'level_{level}_meta.json')
    surv_path = os.path.join(checkpoint_dir, f'level_{level}_survivors.npy')

    np.save(surv_path, survivors)
    metadata['survivor_file'] = surv_path  # backward compatibility
    metadata['checkpoint_survivor_npy'] = surv_path
    metadata.setdefault('storage_mode', 'memory')
    metadata['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def _print_summary(results, total_elapsed, dev):
    """Print final summary."""
    status = results.get("status", "proven" if results.get("proven") else "not_proven")
    reason = results.get("inconclusive_reason")
    log("")
    log("=" * 64)
    log("FINAL SUMMARY")
    log("=" * 64)
    log(f"  Device: {dev}")
    log(f"  STATUS: {status}")
    if reason:
        log(f"  Inconclusive reason: {reason}")
    log(f"  Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    log(f"  c_target: {results['parameters']['c_target']}")
    log(f"  m: {results['parameters']['m']}")
    log(f"  Correction: {results['parameters']['correction']:.6f}")
    log(f"  Threshold T: {results['parameters']['effective_threshold_T']:.6f}")
    log(f"  Levels completed: {len(results['levels'])}")

    for lvl_info in results['levels']:
        lvl = lvl_info['level']
        typ = lvl_info['type']
        surv = lvl_info.get('n_survivors', lvl_info.get('num_parents', '?'))
        t = lvl_info['elapsed']
        log(f"    Level {lvl} ({typ}): {fmt_count(surv)} survivors, {t:.1f}s")

    if status == "proven":
        log("")
        log(f"  FORMALLY PROVEN: c >= {results['proven_bound']}")
        log(f"  (Cloninger-Steinerberger exhaustive verification complete)")
    elif status == "not_proven":
        n_rem = results.get('remaining_survivors', '?')
        log("")
        log(f"  NOT PROVEN: {n_rem} survivors remain")
        log(f"  Options: increase m, add more levels, or reduce c_target")
    else:
        n_rem = results.get('remaining_survivors', '?')
        log("")
        log(f"  INCONCLUSIVE: processing was incomplete")
        log(f"  Remaining survivors (if known): {n_rem}")
        log(f"  No formal proof claim is made for this run.")

    log(f"  Results saved to: {CHECKPOINT_DIR}/")
    log("=" * 64)


if __name__ == "__main__":
    main()
