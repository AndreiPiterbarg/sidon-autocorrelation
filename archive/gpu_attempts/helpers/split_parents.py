#!/usr/bin/env python3
"""Split a parent checkpoint into chunks for multi-GPU processing.

Each chunk is a contiguous slice of the parent array saved as its own .npy file.
If a chunk crashes, only that chunk needs to be re-run.

Usage:
    python split_parents.py parents.npy --chunks 4
    python split_parents.py parents.npy --chunks 40 --output_dir chunks/
    python split_parents.py parents.npy --chunk_size 10000
"""
import argparse
import math
import os

import numpy as np


def split(parents_path, n_chunks=None, chunk_size=None, output_dir=None):
    parents = np.load(parents_path)
    n = len(parents)
    print(f"Loaded {n} parents from {parents_path} (shape: {parents.shape})")

    if chunk_size is not None:
        n_chunks = math.ceil(n / chunk_size)
    elif n_chunks is None:
        n_chunks = 4

    chunk_size = math.ceil(n / n_chunks)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(parents_path), "chunks")
    os.makedirs(output_dir, exist_ok=True)

    manifest = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        if start >= n:
            break
        chunk = parents[start:end]
        fname = f"chunk_{i:04d}.npy"
        fpath = os.path.join(output_dir, fname)
        np.save(fpath, chunk)
        manifest.append(fname)
        print(f"  {fname}: parents [{start}:{end}] ({len(chunk)} rows)")

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        for fname in manifest:
            f.write(fname + "\n")

    print(f"\n{len(manifest)} chunks written to {output_dir}/")
    print(f"Manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Split parent array into chunks")
    parser.add_argument("parents", help="Path to parents .npy file")
    parser.add_argument("--chunks", type=int, default=None, help="Number of chunks")
    parser.add_argument("--chunk_size", type=int, default=None, help="Max parents per chunk")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    split(args.parents, n_chunks=args.chunks, chunk_size=args.chunk_size,
          output_dir=args.output_dir)


if __name__ == "__main__":
    main()
