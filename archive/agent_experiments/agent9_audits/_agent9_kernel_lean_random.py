"""Random soundness check: Numba kernel `prune_P` vs Lean specification.

We sample compositions uniformly over Sₙₘ = {c ∈ ℕ^d : sum c = 4nm} and
compare:
  K = kernel decision (Numba `prune_P`).
  L = Lean decision (real-valued strict `>`).

DESIRED: K is sound w.r.t. L, i.e. ∀ q. K fires at q ⟹ Lean fires at q.
The kernel is, in fact, STRICTER (more conservative): kernel uses
   conv[q] > int(2d·c·m² + 2 W + n_bins + eps)
while Lean uses
   pointeval_value > c_target + correction
   ⟺  (1/(4n)) · conv[q]/m² > c_target + (2W + n_bins)/(4n m²)
   ⟺  conv[q]/m² > 4n·c_target + (2W + n_bins)/m²
   ⟺  conv[q] > 4n·c_target·m² + 2W + n_bins
   ⟺  conv[q] > 2d·c_target·m² + 2W + n_bins

So Lean uses the EXACT real-valued threshold; kernel uses
   conv[q] > floor(2d·c·m² + 2W + n_bins + eps).
Note: 2W + n_bins are integers, c·m² may not be. The kernel adds eps_margin
= 1e-9·m² (tiny positive) before truncating.

For the kernel to be UNSOUND (firing when Lean says no), we'd need
   conv[q] > floor(threshold + eps)  but   conv[q] ≤ threshold.
For integer conv[q] and `floor(threshold + eps) ≤ threshold` (since
eps_margin is small relative to integer gap of 1), this is essentially
impossible UNLESS eps pushes a value from N+0.999... up to N+1.0,
which would lower thr_int from N to N+1 (oops, that's the opposite sign).

Wait — eps_margin INCREASES the threshold; floor of (threshold + eps) is
≥ floor(threshold).  So `conv[q] > floor(thr+eps)` is STRICTER than the
real `conv[q] > thr`.  Therefore:
   kernel fires  ⟹  Lean fires.   [KERNEL is SOUND vs LEAN]

BUT we still want to verify this empirically.  And we want to find any
composition where Lean fires but kernel doesn't (kernel UNDERPRUNES) to
quantify the slack.
"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _M1_bench import prune_P
from _agent9_edge_audit import (lean_pointeval_value, lean_pointeval_correction,
                                  prune_P_ref, lean_i_lo, lean_i_hi,
                                  lean_W_int, lean_n_bins)


def random_composition(d, S, rng):
    """Uniform random composition of S into d parts via stars & bars."""
    bars = sorted(rng.choice(S + d - 1, d - 1, replace=False))
    parts = []
    prev = -1
    for b in bars:
        parts.append(b - prev - 1)
        prev = b
    parts.append(S + d - 1 - prev - 1)
    return parts


def main():
    rng = np.random.default_rng(42)

    n_unsound = 0   # kernel fires, Lean doesn't (must be 0)
    n_underprune = 0  # Lean fires, kernel doesn't (slack)
    n_total = 0
    examples_unsound = []
    examples_underprune = []

    configs = [
        (2, 5, 1.20),  (2, 10, 1.20), (2, 20, 1.20),
        (3, 10, 1.20), (3, 10, 1.28), (3, 20, 1.20),
        (4, 5, 1.20),  (4, 10, 1.20), (4, 10, 1.28),
        (5, 5, 1.28),  (6, 5, 1.20),
        # Tiny d for stress:
        (1, 5, 3.0),   (1, 10, 2.5),  # d=2
    ]

    for n_half, m, ct in configs:
        d = 2 * n_half
        S = 4 * n_half * m
        for trial in range(500):
            c = random_composition(d, S, rng)
            n_total += 1
            # Numba kernel
            arr = np.zeros((1, d), dtype=np.int32)
            arr[0, :] = c
            kernel_pruned = not bool(prune_P(arr, n_half, m, ct)[0])
            # Lean (real-valued, strict >)
            lean_pruned = False
            lean_q = None
            for q in range(2 * d - 1):
                v = lean_pointeval_value(n_half, m, c, q)
                cor = lean_pointeval_correction(n_half, m, c, q)
                if v > ct + cor:
                    lean_pruned = True
                    lean_q = q
                    break

            if kernel_pruned and not lean_pruned:
                n_unsound += 1
                if len(examples_unsound) < 5:
                    examples_unsound.append((n_half, m, ct, c))
            elif lean_pruned and not kernel_pruned:
                n_underprune += 1
                if len(examples_underprune) < 5:
                    examples_underprune.append((n_half, m, ct, c, lean_q))

    print(f"Random soundness check: n_total = {n_total}")
    print(f"  KERNEL UNSOUND vs LEAN (kernel fires, Lean doesn't): {n_unsound}")
    print(f"  KERNEL UNDERPRUNES vs LEAN (Lean fires, kernel doesn't): {n_underprune}")
    if examples_unsound:
        print("\n  ** UNSOUND examples (CRITICAL):")
        for ex in examples_unsound:
            print(f"     {ex}")
    if examples_underprune:
        print("\n  Underprune examples (slack):")
        for ex in examples_underprune:
            print(f"     {ex}")
    print()
    if n_unsound == 0:
        print("  [OK] Kernel is SOUND vs Lean spec across all sampled configs.")


if __name__ == '__main__':
    main()
