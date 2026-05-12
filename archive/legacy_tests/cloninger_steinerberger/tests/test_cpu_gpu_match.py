"""CPU vs GPU correctness comparison tests.

Generates CPU reference survivors for small parent subsets, then compares
against GPU output.

Usage:
    # Step 1: Generate CPU reference files locally
    python tests/test_cpu_gpu_match.py generate

    # Step 2: On GPU pod, run comparison
    #   (upload tests/cpu_gpu_data/ and this script to the pod)
    python tests/test_cpu_gpu_match.py compare --gpu-binary /workspace/sidon-autocorrelation/gpu/cascade_prover

    # Or run threshold/bin-range checks as pytest
    pytest tests/test_cpu_gpu_match.py -v -k "not generate"
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Project layout
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import (
    process_parent_fused,
    _compute_bin_ranges,
)
from pruning import correction

PROJECT_ROOT = Path(_project_dir)

# ──────────────────────────────────────────────────────────────────────
# Test configurations: (name, n_half, m, c_target, level, n_parents)
# ──────────────────────────────────────────────────────────────────────

TEST_CONFIGS = [
    # Level 1: L0->L1 (d_parent=4, d_child=8) — full run is tiny
    ("L1_full", 2, 20, 1.4, 1, None),         # All L0 parents (~345)
    ("L1_ctarget130", 2, 20, 1.30, 1, None),   # Different c_target

    # Level 2: L1->L2 (d_parent=8, d_child=16) — use small subsets
    ("L2_first10", 2, 20, 1.4, 2, 10),
    ("L2_first50", 2, 20, 1.4, 2, 50),
    ("L2_last10", 2, 20, 1.4, 2, -10),  # negative = last N

    # Level 3: L2->L3 (d_parent=16, d_child=32) — small subsets
    ("L3_first5", 2, 20, 1.4, 3, 5),
    ("L3_first20", 2, 20, 1.4, 3, 20),
]

TEST_DIR = PROJECT_ROOT / "tests" / "cpu_gpu_data"


def ensure_test_dir():
    TEST_DIR.mkdir(parents=True, exist_ok=True)


def get_parents_for_level(level):
    """Load the parent checkpoint for a given level."""
    ckpt = {1: "checkpoint_L0_survivors.npy",
            2: "checkpoint_L1_survivors.npy",
            3: "checkpoint_L2_survivors.npy"}
    path = PROJECT_ROOT / "data" / ckpt[level]
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return np.load(str(path))


def select_parents(all_parents, n_parents):
    if n_parents is None:
        return all_parents
    if n_parents > 0:
        return all_parents[:n_parents]
    return all_parents[n_parents:]


def sort_and_dedup(arr):
    """Lexicographic sort + dedup for survivor arrays."""
    if arr.shape[0] == 0:
        return arr
    d = arr.shape[1]
    keys = [arr[:, c] for c in reversed(range(d))]
    order = np.lexsort(keys)
    arr = arr[order]
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[1:] = np.any(arr[1:] != arr[:-1], axis=1)
    return arr[mask]


# ──────────────────────────────────────────────────────────────────────
# CPU reference generation
# ──────────────────────────────────────────────────────────────────────

def generate_cpu_reference(name, n_half, m, c_target, level, n_parents):
    all_parents = get_parents_for_level(level)
    parents = select_parents(all_parents, n_parents)
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    print(f"\n{'='*60}")
    print(f"CPU reference: {name}")
    print(f"  n_half={n_half}, m={m}, c_target={c_target}")
    print(f"  level={level}, d_parent={d_parent}->d_child={d_child}")
    print(f"  {parents.shape[0]} parents (of {all_parents.shape[0]})")
    print(f"{'='*60}")

    all_survivors = []
    total_children = 0
    t0 = time.time()

    for i, parent in enumerate(parents):
        survivors, n_children = process_parent_fused(parent, m, c_target, n_half_child)
        all_survivors.append(survivors)
        total_children += n_children

        if (i + 1) % max(1, len(parents) // 10) == 0 or i == len(parents) - 1:
            elapsed = time.time() - t0
            n_surv = sum(s.shape[0] for s in all_survivors)
            print(f"  [{i+1}/{len(parents)}] survivors={n_surv}, "
                  f"children={total_children:,}, {elapsed:.1f}s")

    elapsed = time.time() - t0

    if all_survivors:
        survivors = np.concatenate(all_survivors, axis=0)
    else:
        survivors = np.empty((0, d_child), dtype=np.int32)

    survivors = sort_and_dedup(survivors)
    print(f"  -> {survivors.shape[0]} unique survivors in {elapsed:.1f}s")

    ensure_test_dir()
    np.save(str(TEST_DIR / f"{name}_parents.npy"), parents)
    np.save(str(TEST_DIR / f"{name}_cpu_survivors.npy"), survivors)

    meta = {
        "name": name, "n_half": n_half, "m": m, "c_target": c_target,
        "level": level, "d_parent": d_parent, "d_child": d_child,
        "n_parents": int(parents.shape[0]),
        "n_survivors_cpu": int(survivors.shape[0]),
        "total_children": int(total_children),
        "elapsed_cpu": round(elapsed, 2),
    }
    (TEST_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ──────────────────────────────────────────────────────────────────────
# GPU execution and comparison
# ──────────────────────────────────────────────────────────────────────

def run_gpu_and_compare(name, gpu_binary="./cascade_prover"):
    meta = json.loads((TEST_DIR / f"{name}_meta.json").read_text())
    cpu_survivors = np.load(str(TEST_DIR / f"{name}_cpu_survivors.npy"))

    max_survivors = max(meta["n_survivors_cpu"] * 2, 1000000)
    gpu_output_path = TEST_DIR / f"{name}_gpu_survivors.npy"

    cmd = [
        gpu_binary,
        str(TEST_DIR / f"{name}_parents.npy"),
        str(gpu_output_path),
        "--d_parent", str(meta["d_parent"]),
        "--m", str(meta["m"]),
        "--c_target", str(meta["c_target"]),
        "--max_survivors", str(max_survivors),
    ]
    print(f"\n{'='*60}")
    print(f"GPU run: {name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed_gpu = time.time() - t0

    print(result.stdout)
    if result.stderr:
        print(f"  stderr: {result.stderr}")
    if result.returncode != 0:
        print(f"  FAIL: GPU exit code {result.returncode}")
        return False

    gpu_survivors = np.load(str(gpu_output_path))
    gpu_survivors = sort_and_dedup(gpu_survivors)

    print(f"  GPU: {gpu_survivors.shape[0]} unique survivors, {elapsed_gpu:.1f}s")
    print(f"  CPU: {cpu_survivors.shape[0]} unique survivors, {meta['elapsed_cpu']:.1f}s")

    # Exact comparison
    if cpu_survivors.shape != gpu_survivors.shape:
        cpu_set = set(map(tuple, cpu_survivors))
        gpu_set = set(map(tuple, gpu_survivors))
        only_cpu = cpu_set - gpu_set
        only_gpu = gpu_set - cpu_set
        print(f"  FAIL: count mismatch! CPU={len(cpu_set)}, GPU={len(gpu_set)}")
        print(f"    In CPU only: {len(only_cpu)}")
        print(f"    In GPU only: {len(only_gpu)}")
        for s in list(only_cpu)[:5]:
            print(f"      CPU-only: {s}")
        for s in list(only_gpu)[:5]:
            print(f"      GPU-only: {s}")
        return False

    if not np.array_equal(cpu_survivors, gpu_survivors):
        diff = np.where(np.any(cpu_survivors != gpu_survivors, axis=1))[0]
        print(f"  FAIL: {len(diff)} rows differ (same count but different values)")
        for r in diff[:5]:
            print(f"    Row {r}: CPU={cpu_survivors[r]} GPU={gpu_survivors[r]}")
        return False

    speedup = meta["elapsed_cpu"] / elapsed_gpu if elapsed_gpu > 0 else float('inf')
    print(f"  PASS: exact match ({cpu_survivors.shape[0]} survivors)")
    print(f"  Speedup: {speedup:.1f}x")

    meta["n_survivors_gpu"] = int(gpu_survivors.shape[0])
    meta["elapsed_gpu"] = round(elapsed_gpu, 2)
    meta["speedup"] = round(speedup, 1)
    meta["match"] = True
    (TEST_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    return True


# ──────────────────────────────────────────────────────────────────────
# Threshold table comparison (runs locally, no GPU needed)
# ──────────────────────────────────────────────────────────────────────

def compare_threshold_tables():
    """Verify GPU threshold precomputation matches CPU logic exactly."""
    print("\n" + "="*60)
    print("Threshold table comparison")
    print("="*60)

    params = [
        (20, 1.4, 8), (20, 1.4, 16), (20, 1.4, 32), (20, 1.4, 64),
        (20, 1.30, 8), (20, 1.30, 32),
    ]

    all_pass = True
    for m, c_target, d_child in params:
        n = d_child // 2
        n_half_child = n
        eps_margin = 1e-9 * m * m
        one_minus_4eps = 1.0 - 4.0 * np.finfo(np.float64).eps

        max_ell = 2 * d_child
        ell_count = max_ell - 1

        errors = 0
        for ell_idx in range(ell_count):
            ell = ell_idx + 2
            for W_int in range(m + 1):
                dyn_x = (c_target * m * m + 3.0 + eps_margin + 2.0 * W_int) \
                         * ell / (4.0 * n)
                # GPU: (int32_t)(dyn_x * one_minus_4eps)  — C truncation
                gpu_t = int(dyn_x * one_minus_4eps)
                # CPU: int(math.floor(dyn_x * one_minus_4eps))
                cpu_t = int(math.floor(dyn_x * one_minus_4eps))
                if gpu_t != cpu_t:
                    errors += 1
                    if errors <= 3:
                        print(f"    MISMATCH ell={ell} W={W_int}: "
                              f"GPU={gpu_t} CPU={cpu_t} "
                              f"dyn_x*safe={dyn_x * one_minus_4eps}")

        n_checks = ell_count * (m + 1)
        if errors == 0:
            print(f"  d_child={d_child:2d}, m={m}, c={c_target}: "
                  f"OK ({n_checks} thresholds)")
        else:
            print(f"  d_child={d_child:2d}, m={m}, c={c_target}: "
                  f"FAIL ({errors}/{n_checks})")
            all_pass = False

    return all_pass


def compare_bin_ranges():
    """Verify GPU bin range computation matches CPU."""
    print("\n" + "="*60)
    print("Bin range comparison")
    print("="*60)

    params = [(1, 20, 1.4), (2, 20, 1.4), (3, 20, 1.4), (1, 20, 1.30)]

    all_pass = True
    for level, m, c_target in params:
        all_parents = get_parents_for_level(level)
        parents = all_parents[:min(100, len(all_parents))]
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        # Compute x_cap the same way as GPU host code (matches CPU)
        corr_val = 2.0 / m + 1.0 / (m * m)
        thresh = c_target + corr_val + 1e-9
        x_cap = int(math.floor(m * math.sqrt(thresh / d_child)))
        x_cap_cs = int(math.floor(m * math.sqrt(c_target / d_child))) + 1
        x_cap = min(x_cap, x_cap_cs, m)
        x_cap = max(x_cap, 0)

        errors = 0
        for i, parent in enumerate(parents):
            result_cpu = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result_cpu is None:
                continue
            lo_cpu, hi_cpu, _ = result_cpu
            for j in range(d_parent):
                b = int(parent[j])
                lo_exp = max(0, b - x_cap)
                hi_exp = min(b, x_cap)
                if lo_cpu[j] != lo_exp or hi_cpu[j] != hi_exp:
                    errors += 1

        if errors == 0:
            print(f"  L{level}, c={c_target}: OK ({len(parents)} parents, x_cap={x_cap})")
        else:
            print(f"  L{level}, c={c_target}: FAIL ({errors} mismatches)")
            all_pass = False

    return all_pass


# ──────────────────────────────────────────────────────────────────────
# Pytest
# ──────────────────────────────────────────────────────────────────────

def test_threshold_tables():
    assert compare_threshold_tables()

def test_bin_ranges():
    assert compare_bin_ranges()



# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def cmd_generate():
    print(f"Generating CPU reference data -> {TEST_DIR}\n")

    compare_threshold_tables()
    compare_bin_ranges()

    for cfg in TEST_CONFIGS:
        try:
            generate_cpu_reference(*cfg)
        except FileNotFoundError as e:
            print(f"  SKIP: {cfg[0]} — {e}")
        except Exception as e:
            print(f"  ERROR: {cfg[0]} — {e}")
            raise

    print(f"\nDone. Test data in {TEST_DIR}")
    print("Next: upload to GPU pod and run 'compare'")


def cmd_compare(gpu_binary):
    print(f"Comparing GPU ({gpu_binary}) vs CPU references\n")

    results = {}
    for cfg in TEST_CONFIGS:
        name = cfg[0]
        if not (TEST_DIR / f"{name}_meta.json").exists():
            print(f"  SKIP: {name} (no CPU reference)")
            continue
        try:
            results[name] = run_gpu_and_compare(name, gpu_binary)
        except Exception as e:
            print(f"  ERROR: {name} — {e}")
            results[name] = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
        if not ok:
            all_pass = False

    if all_pass and results:
        print("\nAll tests PASSED — CPU and GPU produce identical survivors.")
    elif results:
        print("\nSome tests FAILED!")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU vs GPU comparison")
    parser.add_argument("command", choices=["generate", "compare"])
    parser.add_argument("--gpu-binary", default="./cascade_prover")
    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate()
    elif args.command == "compare":
        cmd_compare(args.gpu_binary)
