"""Compare GPU checkpoint output with CPU reference."""
import sys
import numpy as np


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


def main():
    gpu_path = sys.argv[1]
    cpu_path = sys.argv[2]

    gpu = np.load(gpu_path)
    cpu = np.load(cpu_path)

    print(f"GPU raw: {gpu.shape}")
    print(f"CPU ref: {cpu.shape}")

    gpu = sort_and_dedup(gpu)
    cpu = sort_and_dedup(cpu)

    print(f"GPU unique: {gpu.shape[0]}")
    print(f"CPU unique: {cpu.shape[0]}")

    if gpu.shape != cpu.shape:
        print(f"FAIL: shape mismatch")
        cpu_set = set(map(tuple, cpu))
        gpu_set = set(map(tuple, gpu))
        print(f"  CPU-only: {len(cpu_set - gpu_set)}")
        print(f"  GPU-only: {len(gpu_set - cpu_set)}")
        sys.exit(1)

    if np.array_equal(cpu, gpu):
        print("PASS: exact match")
    else:
        diff = np.where(np.any(cpu != gpu, axis=1))[0]
        print(f"FAIL: {len(diff)} rows differ")
        sys.exit(1)


if __name__ == "__main__":
    main()
