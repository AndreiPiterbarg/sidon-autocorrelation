"""Empirical audit of prune_D vs prune_A.

Run prune_A and prune_D on a 1000-composition batch with
(n_half, m, c_target) = (3, 10, 1.20).  Verify:
  - sD <= sA pointwise (D's survivors are a subset of A's).
  - Pruning counts.

Also include L3 sanity check: ell_int_arr[k] / (4n)  vs  1/2 - |x_k|.
"""
import os
import sys

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from _stage1_bench import prune_A, prune_D
from compositions import generate_compositions_batched


# -----------------------------------------------------------
# L3 sanity check: ell_int_arr[k] / (4n)  vs  1/2 - |x_k|
# -----------------------------------------------------------
def l3_check(n_half=2):
    d = 2 * n_half
    conv_len = 2 * d - 1
    two_n = 2 * n_half
    print(f"\nL3 check: n_half={n_half}, d={d}, conv_len={conv_len}, 4n={4*n_half}")
    print("  k    x_k=(k+1)/(4n)-1/2    1/2-|x_k|    ell_int/4n   match?")
    for k in range(conv_len):
        d_idx = abs((k + 1) - two_n)
        v = max(0, two_n - d_idx)
        x_k = (k + 1) / (4 * n_half) - 0.5
        L_xk = max(0.0, 0.5 - abs(x_k))
        ratio = v / (4 * n_half)
        match = abs(L_xk - ratio) < 1e-12
        print(f"  {k:2d}     {x_k:+.4f}             {L_xk:.4f}        "
              f"{ratio:.4f}      {match}")


# -----------------------------------------------------------
# Empirical run (3, 10, 1.20) on a 1000-composition batch
# -----------------------------------------------------------
def empirical(n_half=3, m=10, c_target=1.20, batch_size=1000):
    d = 2 * n_half
    S_half = 2 * n_half * m
    print(f"\n=== Empirical: n_half={n_half}, m={m}, c_target={c_target}"
          f", batch_size={batch_size} ===")
    print(f"d={d}, palindromic half_sum={S_half}")

    # Warm up JIT
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_A(warm, n_half, m, c_target)
    prune_D(warm, n_half, m, c_target)

    # Pull a single batch of up to `batch_size` palindromic compositions
    half_iter = generate_compositions_batched(n_half, S_half,
                                                batch_size=batch_size)
    half_batch = next(iter(half_iter))
    if half_batch.shape[0] > batch_size:
        half_batch = half_batch[:batch_size]
    batch = np.empty((half_batch.shape[0], d), dtype=np.int32)
    batch[:, :n_half] = half_batch
    batch[:, n_half:] = half_batch[:, ::-1]

    sA = prune_A(batch, n_half, m, c_target)
    sD = prune_D(batch, n_half, m, c_target)
    pruned_A = int(np.sum(~sA))
    pruned_D = int(np.sum(~sD))
    survivors_A = int(np.sum(sA))
    survivors_D = int(np.sum(sD))
    soundness = bool(np.all(sD <= sA))
    extra_prune = int(np.sum(sA & ~sD))
    print(f"  A:  pruned={pruned_A:4d}, survivors={survivors_A:4d}")
    print(f"  D:  pruned={pruned_D:4d}, survivors={survivors_D:4d}")
    print(f"  D <= A pointwise (D survivors subset of A's)? {soundness}")
    print(f"  Compositions pruned by D but not A: "
          f"{int(np.sum(~sD & sA))}  (D extra pruning)")
    print(f"  Compositions pruned by A but not D: "
          f"{int(np.sum(~sA & sD))}  (should be 0 if D dominates A)")
    if survivors_A:
        print(f"  D removes {extra_prune}/{survivors_A} = "
              f"{100.0 * extra_prune / survivors_A:.2f}% of A's survivors")


if __name__ == '__main__':
    l3_check(n_half=2)
    l3_check(n_half=3)
    empirical(n_half=3, m=10, c_target=1.20, batch_size=1000)
