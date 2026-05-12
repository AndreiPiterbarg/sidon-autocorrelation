"""Standalone GPU comparison script — runs on the pod, no CPU dependencies.

Loads pre-generated CPU reference survivors, runs GPU binary, compares.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_DIR = SCRIPT_DIR / "cpu_gpu_data"


def sort_and_dedup(arr):
    if arr.shape[0] == 0:
        return arr
    d = arr.shape[1]
    keys = [arr[:, c] for c in reversed(range(d))]
    order = np.lexsort(keys)
    arr = arr[order]
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[1:] = np.any(arr[1:] != arr[:-1], axis=1)
    return arr[mask]


def run_gpu_and_compare(name, gpu_binary):
    meta_path = TEST_DIR / f"{name}_meta.json"
    if not meta_path.exists():
        print(f"  SKIP: {name} (no meta file)")
        return None

    meta = json.loads(meta_path.read_text())
    cpu_survivors = np.load(str(TEST_DIR / f"{name}_cpu_survivors.npy"))

    max_survivors = max(meta["n_survivors_cpu"] * 2, 1000000)
    gpu_output = TEST_DIR / f"{name}_gpu_survivors.npy"

    cmd = [
        gpu_binary,
        str(TEST_DIR / f"{name}_parents.npy"),
        str(gpu_output),
        "--d_parent", str(meta["d_parent"]),
        "--m", str(meta["m"]),
        "--c_target", str(meta["c_target"]),
        "--max_survivors", str(max_survivors),
    ]

    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  d_parent={meta['d_parent']}, d_child={meta['d_child']}, "
          f"m={meta['m']}, c_target={meta['c_target']}")
    print(f"  {meta['n_parents']} parents, CPU survivors={meta['n_survivors_cpu']}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed_gpu = time.time() - t0

    # Print GPU output
    for line in result.stdout.strip().split('\n'):
        print(f"  GPU| {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().split('\n'):
            print(f"  ERR| {line}")

    if result.returncode != 0:
        print(f"  FAIL: GPU exit code {result.returncode}")
        return False

    # Load and sort GPU output
    gpu_survivors = np.load(str(gpu_output))
    gpu_survivors = sort_and_dedup(gpu_survivors)

    print(f"\n  GPU unique survivors: {gpu_survivors.shape[0]} ({elapsed_gpu:.2f}s)")
    print(f"  CPU unique survivors: {cpu_survivors.shape[0]} ({meta['elapsed_cpu']:.2f}s)")

    # Exact comparison
    if cpu_survivors.shape[0] != gpu_survivors.shape[0]:
        cpu_set = set(map(tuple, cpu_survivors))
        gpu_set = set(map(tuple, gpu_survivors))
        only_cpu = cpu_set - gpu_set
        only_gpu = gpu_set - cpu_set
        print(f"\n  FAIL: survivor count mismatch!")
        print(f"    CPU-only: {len(only_cpu)}")
        print(f"    GPU-only: {len(only_gpu)}")
        for s in list(only_cpu)[:3]:
            print(f"      CPU-only: {list(s)}")
        for s in list(only_gpu)[:3]:
            print(f"      GPU-only: {list(s)}")
        return False

    if not np.array_equal(cpu_survivors, gpu_survivors):
        diff = np.where(np.any(cpu_survivors != gpu_survivors, axis=1))[0]
        print(f"\n  FAIL: {len(diff)} rows differ")
        for r in diff[:3]:
            print(f"    Row {r}: CPU={list(cpu_survivors[r])}")
            print(f"             GPU={list(gpu_survivors[r])}")
        return False

    speedup = meta["elapsed_cpu"] / elapsed_gpu if elapsed_gpu > 0 else float('inf')
    print(f"\n  PASS: exact match ({cpu_survivors.shape[0]} survivors)")
    if meta["elapsed_cpu"] > 0.01:
        print(f"  Speedup: {speedup:.1f}x")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-binary", required=True)
    parser.add_argument("--test", default=None, help="Run only this test")
    args = parser.parse_args()

    # Discover all tests from meta files
    test_names = []
    for f in sorted(TEST_DIR.glob("*_meta.json")):
        name = f.stem.replace("_meta", "")
        if args.test and args.test != name:
            continue
        test_names.append(name)

    if not test_names:
        print("No test cases found!")
        sys.exit(1)

    print(f"Running {len(test_names)} GPU comparison tests")
    print(f"GPU binary: {args.gpu_binary}")

    results = {}
    for name in test_names:
        try:
            results[name] = run_gpu_and_compare(name, args.gpu_binary)
        except Exception as e:
            print(f"  ERROR: {name} -- {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        if ok is None:
            print(f"  SKIP: {name}")
        elif ok:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name}")
            all_pass = False

    if all_pass and any(v is True for v in results.values()):
        print("\nAll tests PASSED - CPU and GPU produce identical survivors.")
    elif any(v is False for v in results.values()):
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
