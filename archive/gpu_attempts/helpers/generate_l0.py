#!/usr/bin/env python3
"""Generate L0 checkpoint for the GPU cascade prover.

This runs the CPU L0 enumeration (fast — seconds for any m) and saves
the survivors as a .npy file that the GPU cascade_prover can read.

Usage:
    python generate_l0.py --m 35 --c_target 1.33
    python generate_l0.py --m 35 --c_target 1.33 --output data/checkpoint_L0_survivors.npy
"""
import argparse
import os
import sys

import numpy as np

# Path setup: find the cloninger-steinerberger package
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
sys.path.insert(0, os.path.join(_project_root, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_project_root, 'cloninger-steinerberger', 'cpu'))

from run_cascade import run_level0


def main():
    parser = argparse.ArgumentParser(description='Generate L0 checkpoint for GPU prover')
    parser.add_argument('--m', type=int, required=True, help='Mass parameter')
    parser.add_argument('--c_target', type=float, required=True, help='Target constant')
    parser.add_argument('--n_half', type=int, default=2, help='n_half for L0 (default: 2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: ../data/checkpoint_L0_survivors.npy)')
    args = parser.parse_args()

    if args.output is None:
        data_dir = os.path.join(_project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        args.output = os.path.join(data_dir, 'checkpoint_L0_survivors.npy')

    print(f"Generating L0 checkpoint: m={args.m}, c_target={args.c_target}, n_half={args.n_half}")

    result = run_level0(n_half=args.n_half, m=args.m, c_target=args.c_target, verbose=True)
    survivors = result['survivors']

    print(f"\nL0 survivors: {len(survivors)} (shape: {survivors.shape})")

    np.save(args.output, survivors)
    print(f"Saved to: {args.output}")

    # Rigorousness check
    d_child = 2 * survivors.shape[1]  # L0 survivors become parents for L1
    max_rigorous_d = args.m
    print(f"\nRigorousness: d_child at L1 = {d_child}, m = {args.m}")
    if d_child <= max_rigorous_d:
        print(f"  L1 is RIGOROUS (d={d_child} <= m={args.m})")
    else:
        print(f"  L1 is NOT RIGOROUS (d={d_child} > m={args.m})")


if __name__ == '__main__':
    main()
