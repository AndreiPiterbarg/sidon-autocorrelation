"""Push S to the maximum that any cascade level uses.  Standard production:
n_half=2, m=20, c_target=32/25.  Cascade refines via doubling (n_half → 2*n_half).
At depth k, n_half = 2 + k? — actually depends on cascade design.  Common:
n_half = 2, 4, 8, 16, ...  Refining keeps S = 4 n_half m fixed.

Worst case in CascadePrunedP refine: arbitrary depth.  At n_half=64, m=20,
S = 4·64·20 = 5120.  conv[q] up to 2·S² = 52M.  All within int64.

Test compositions where the int(thr) accumulator might lose precision:
  thr = 2·d·c·m² + 2W + n_bins + eps_margin
  At max d=128 (n_half=64), m=20: 2·128·1.28·400 = 131072.  Plus W up to S = 5120.
  thr ≤ 131072 + 2·5120 + 128 + tiny = 141440.  Trivially in float64 exact.
  conv[q] up to 2 S² ≈ 52M; also in float64 exact (mantissa 52 bits).
  No precision issue.

However, at LARGER c_target·m² interactions, double precision could fail
near 2^53 ≈ 9e15.  c_target·m² with m up to 1024 (extreme), m²=1M, ct·m²
= up to 1.28M.  At n_half = 1e9, this is 5e15 — close to 2^53.  In practice
the project uses n_half ≤ 24 or so.  Fine.

The actual production cascade goes n_half = 2, 4, 8, 16, 32, depending on
config.  Let's stress at n_half = 32, m = 20.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _M1_bench import prune_P


def stress(n_half, m, c_target):
    d = 2 * n_half
    S = 4 * n_half * m
    print(f"  n_half={n_half}, d={d}, m={m}, S={S}, c_target={c_target}")
    # All-mass-bin0 (max conv[0])
    c = [S] + [0] * (d - 1)
    arr = np.zeros((1, d), dtype=np.int32)
    arr[0, :] = c
    pruned = not bool(prune_P(arr, n_half, m, c_target)[0])
    print(f"    [S, 0, ..., 0]: pruned={pruned} (expected True; conv[0]=S²={S*S}, base={2*d*c_target*m*m})")
    # Uniform 2m
    c = [2 * m] * d
    arr[0, :] = c
    pruned = not bool(prune_P(arr, n_half, m, c_target)[0])
    print(f"    uniform 2m={2*m}: pruned={pruned} (expected; conv[d-1]={d*(2*m)**2})")


print("=== EXTREME S STRESS ===")
for n_half in [4, 8, 16, 24, 32]:
    stress(n_half, 20, 1.28)
    stress(n_half, 64, 1.28)
print()
print("Max S tested: 4*32*64 =", 4*32*64)
