#!/usr/bin/env python3
"""Merge and deduplicate survivor arrays from chunked GPU runs.

Reads all output_*.npy files from a directory, concatenates, deduplicates,
and saves the merged result.

Usage:
    python merge_survivors.py results/ --output checkpoint_L3_survivors.npy
    python merge_survivors.py results/    # auto-names output
"""
import argparse
import glob
import os

import numpy as np


def merge(results_dir, output_path=None):
    # Find all output files
    patterns = [
        os.path.join(results_dir, "output_*.npy"),
        os.path.join(results_dir, "survivors_*.npy"),
    ]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))

    if not files:
        print(f"No output_*.npy or survivors_*.npy files found in {results_dir}")
        return None

    print(f"Found {len(files)} result files:")
    arrays = []
    total = 0
    for f in files:
        arr = np.load(f)
        print(f"  {os.path.basename(f)}: {len(arr)} survivors")
        if len(arr) > 0:
            arrays.append(arr)
            total += len(arr)

    if not arrays:
        print("\nNo survivors across all chunks — proof COMPLETE at this level!")
        if output_path:
            np.save(output_path, np.empty((0, arrays[0].shape[1] if arrays else 0), dtype=np.int32))
        return np.empty((0,), dtype=np.int32)

    merged = np.vstack(arrays)
    print(f"\nTotal before dedup: {total}")

    # Deduplicate
    merged = np.unique(merged, axis=0)
    print(f"Total after dedup:  {len(merged)}")

    if output_path is None:
        output_path = os.path.join(results_dir, "merged_survivors.npy")

    np.save(output_path, merged)
    print(f"Saved to: {output_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge chunked survivor outputs")
    parser.add_argument("results_dir", help="Directory containing output_*.npy files")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    merge(args.results_dir, output_path=args.output)


if __name__ == "__main__":
    main()
