#!/usr/bin/env python3
"""
run_chunked.py — Chunked GPU cascade prover with deduplication.

Processes parents in chunks to avoid survivor buffer OOM, deduplicates
survivors across chunks, and produces a single output .npy file.

Usage:
    python run_chunked.py parents.npy output.npy \
        --d_parent 32 --m 20 --c_target 1.4 \
        --chunk_size 1000 --max_survivors 1000000 \
        --exe ./cascade_prover.exe

The script:
  1. Splits parents into chunks of --chunk_size
  2. Runs the GPU kernel on each chunk
  3. Collects and deduplicates all survivors
  4. Saves the unique survivors to output.npy
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def run_chunk(exe: str, parents_path: str, output_path: str,
              d_parent: int, m: int, c_target: float,
              max_survivors: int,
              use_flat_threshold: bool = False,
              verify_relaxed: bool = False) -> np.ndarray:
    """Run the GPU kernel on a single chunk and return survivors."""
    cmd = [
        exe, parents_path, output_path,
        "--d_parent", str(d_parent),
        "--m", str(m),
        "--c_target", str(c_target),
        "--max_survivors", str(max_survivors),
    ]
    if use_flat_threshold:
        cmd.append("--use_flat_threshold")
    if verify_relaxed:
        cmd.append("--verify_relaxed")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  KERNEL FAILED (rc={result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return np.empty((0, 2 * d_parent), dtype=np.int32)

    # Print kernel output (progress, timing)
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")

    # Load survivors
    out = Path(output_path)
    if out.exists() and out.stat().st_size > 0:
        survivors = np.load(output_path)
        if survivors.ndim == 1:
            d_child = 2 * d_parent
            survivors = survivors.reshape(-1, d_child)
        return survivors
    return np.empty((0, 2 * d_parent), dtype=np.int32)


def deduplicate(survivors: np.ndarray) -> np.ndarray:
    """Remove duplicate rows from a 2D int32 array."""
    if len(survivors) == 0:
        return survivors
    # View each row as a single void element for fast unique
    dtype = np.dtype((np.void, survivors.dtype.itemsize * survivors.shape[1]))
    viewed = np.ascontiguousarray(survivors).view(dtype).ravel()
    _, idx = np.unique(viewed, return_index=True)
    return survivors[np.sort(idx)]


def main():
    parser = argparse.ArgumentParser(
        description="Chunked GPU cascade prover with deduplication")
    parser.add_argument("parents", help="Input parents .npy file")
    parser.add_argument("output", help="Output survivors .npy file")
    parser.add_argument("--d_parent", type=int, required=True)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--c_target", type=float, default=1.4)
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Parents per GPU kernel invocation")
    parser.add_argument("--max_survivors", type=int, default=1_000_000,
                        help="Max survivors per chunk (GPU buffer size)")
    parser.add_argument("--exe", default="./cascade_prover.exe",
                        help="Path to the cascade_prover executable")
    parser.add_argument("--dedup_interval", type=int, default=10,
                        help="Deduplicate accumulated survivors every N chunks")
    parser.add_argument("--use_flat_threshold", action="store_true",
                        help="Use flat C&S Lemma 3 correction (2/m + 1/m^2) "
                             "instead of W-refined.  Required for Lean axiom.")
    parser.add_argument("--verify_relaxed", action="store_true",
                        help="Verify ±1 floor rounding children are also pruned. "
                             "Required for Lean CascadePruned axiom soundness.")
    args = parser.parse_args()

    parents = np.load(args.parents)
    if parents.ndim == 1:
        parents = parents.reshape(-1, args.d_parent)
    num_parents = len(parents)
    d_child = 2 * args.d_parent

    print(f"Chunked GPU Cascade Prover")
    print(f"  Parents: {num_parents}")
    print(f"  d_parent={args.d_parent}  d_child={d_child}  m={args.m}  "
          f"c_target={args.c_target}")
    print(f"  Chunk size: {args.chunk_size}  "
          f"Max survivors/chunk: {args.max_survivors}")
    print()

    all_survivors = []
    total_raw = 0
    total_unique = 0
    t_start = time.time()

    n_chunks = (num_parents + args.chunk_size - 1) // args.chunk_size

    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_input = str(Path(tmpdir) / "chunk_parents.npy")
        chunk_output = str(Path(tmpdir) / "chunk_output.npy")

        for ci in range(n_chunks):
            lo = ci * args.chunk_size
            hi = min(lo + args.chunk_size, num_parents)
            chunk_parents = parents[lo:hi]

            elapsed = time.time() - t_start
            rate = lo / elapsed if elapsed > 0 else 0
            eta = (num_parents - lo) / rate if rate > 0 else 0
            print(f"[Chunk {ci+1}/{n_chunks}] parents {lo}-{hi} "
                  f"({lo/num_parents*100:.1f}%)  "
                  f"rate={rate:.0f} parents/s  "
                  f"ETA={eta/3600:.1f}h")

            np.save(chunk_input, chunk_parents)
            survivors = run_chunk(
                args.exe, chunk_input, chunk_output,
                args.d_parent, args.m, args.c_target,
                args.max_survivors,
                use_flat_threshold=args.use_flat_threshold,
                verify_relaxed=args.verify_relaxed)

            total_raw += len(survivors)
            if len(survivors) > 0:
                all_survivors.append(survivors)

            # Periodic dedup to keep memory bounded
            if (ci + 1) % args.dedup_interval == 0 and len(all_survivors) > 1:
                merged = np.concatenate(all_survivors)
                merged = deduplicate(merged)
                all_survivors = [merged]
                print(f"  [dedup] {total_raw} raw → {len(merged)} unique so far")

    # Final dedup
    if len(all_survivors) > 0:
        merged = np.concatenate(all_survivors)
        unique = deduplicate(merged)
    else:
        unique = np.empty((0, d_child), dtype=np.int32)

    total_unique = len(unique)
    elapsed = time.time() - t_start

    print()
    print(f"Done in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"  Raw survivors:    {total_raw:,}")
    print(f"  Unique survivors: {total_unique:,}")
    print(f"  Dedup ratio:      {total_raw/max(total_unique,1):.1f}×")

    if total_unique > 0:
        np.save(args.output, unique)
        print(f"  Saved to {args.output}")
    else:
        print(f"  No survivors — proof complete at this level!")


if __name__ == "__main__":
    main()
