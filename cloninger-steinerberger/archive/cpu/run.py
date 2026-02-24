"""Run the Cloninger-Steinerberger branch-and-prune algorithm.

Demonstrates proving lower bounds on C_{1a} at various (n, m) settings.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from core import (
    run_single_level, find_best_bound_direct, correction, count_compositions,
    compute_test_value_single, asymmetry_threshold,
)
import numpy as np


def main():
    print("=" * 70)
    print("CLONINGER-STEINERBERGER BRANCH-AND-PRUNE")
    print("Lower bounds on C_{1a} via Lemma 1 + discretization")
    print("=" * 70)

    # =========================================================================
    # Phase 0: Verify test values for known configurations
    # =========================================================================
    print("\n--- Phase 0: Sanity checks ---\n")

    # Uniform config at n=2: a = (2, 2, 2, 2), sum = 8
    tv_uniform = compute_test_value_single([2, 2, 2, 2], n_half=2)
    print(f"  n=2 uniform (2,2,2,2): test_val = {tv_uniform:.6f} (expect ~1.25)")

    # Uniform config at n=3: a = (2, 2, 2, 2, 2, 2), sum = 12
    tv_uniform3 = compute_test_value_single([2, 2, 2, 2, 2, 2], n_half=3)
    print(f"  n=3 uniform (2,2,2,2,2,2): test_val = {tv_uniform3:.6f} (expect ~1.333)")

    # One-sided config at n=2: a = (4, 4, 0, 0)
    tv_onesided = compute_test_value_single([4, 4, 0, 0], n_half=2)
    print(f"  n=2 one-sided (4,4,0,0): test_val = {tv_onesided:.6f} (expect >> 1.28)")

    # Symmetric non-uniform at n=2: a = (2.5, 1.5, 1.5, 2.5)
    tv_sym = compute_test_value_single([2.5, 1.5, 1.5, 2.5], n_half=2)
    print(f"  n=2 symmetric (2.5,1.5,1.5,2.5): test_val = {tv_sym:.6f}")

    # Asymmetry threshold for target 1.28
    at = asymmetry_threshold(1.28)
    print(f"\n  Asymmetry threshold for target 1.28: left_frac >= {at:.4f}")
    print(f"  Asymmetry threshold for target 1.00: left_frac >= "
          f"{asymmetry_threshold(1.0):.4f}")

    # =========================================================================
    # Phase 1: Compute best provable bounds via single-pass direct method
    # =========================================================================
    print("\n\n--- Phase 1: Best provable bounds (single-pass direct) ---\n")

    for n_half, m in [(2, 10), (2, 20), (2, 50), (3, 3), (3, 5)]:
        d = 2 * n_half
        S = m  # S=m convention
        n_configs = count_compositions(d, S)
        corr = correction(m)
        print(f"\n  n={n_half}, m={m}: {n_configs:,} configs, "
              f"correction={corr:.4f}")

        if n_configs > 50_000_000:
            print(f"  Skipping (too many configs for quick run)")
            continue

        bound = find_best_bound_direct(n_half, m, verbose=True)
        print(f"  PROVEN C_{{1a}} >= {bound:.6f}")

    # =========================================================================
    # Phase 2: Push for best bound with larger computation
    # =========================================================================
    print("\n\n--- Phase 2: Pushing for best bound ---\n")

    for n_half, m in [(2, 100), (2, 200)]:
        d = 2 * n_half
        S = m  # S=m convention
        n_configs = count_compositions(d, S)
        print(f"  n={n_half}, m={m}: {n_configs:,} configs")
        if n_configs > 500_000_000:
            print(f"  Skipping (too many)")
            continue

        bound = find_best_bound_direct(n_half, m, verbose=True)
        print(f"  PROVEN C_{{1a}} >= {bound:.6f}")

    # n=3 with m=10
    n_half, m = 3, 10
    d = 2 * n_half
    S = m  # S=m convention
    n_configs = count_compositions(d, S)
    print(f"\n  n={n_half}, m={m}: {n_configs:,} configs")
    if n_configs <= 500_000_000:
        bound = find_best_bound_direct(n_half, m, verbose=True)
        print(f"  PROVEN C_{{1a}} >= {bound:.6f}")
    else:
        print(f"  Skipping (too many)")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The Cloninger-Steinerberger algorithm proves rigorous lower bounds on C_{1a}
by exhaustive case analysis over discretized mass distributions.

The single-pass direct method computes the exact best provable bound in one
enumeration pass (~8-10x faster than binary search), by tracking:
  bound = min_b max(test_val(b) - correction, asym_bound(b))
""")


if __name__ == '__main__':
    main()
