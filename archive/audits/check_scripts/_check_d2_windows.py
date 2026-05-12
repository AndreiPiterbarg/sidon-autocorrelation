"""Quick check: list d=2 windows and L0 cells at (n=1, m=20, c=1.281)."""
import sys, os
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

import numpy as np
from _Q_bench import _build_windows, _enum_balanced_signs
from compositions import generate_compositions_batched
from pruning import count_compositions

n_half = 1
m = 20
c_target = 1.281
d = 2 * n_half  # = 2
S_full = 4 * n_half * m  # = 80
S_half = 2 * n_half * m  # = 40

print(f"d={d}, S_full={S_full}, S_half={S_half}")

windows, ell_int_sums = _build_windows(d)
print(f"n_windows = {len(windows)}")
for w, e in zip(windows, ell_int_sums):
    print(f"  ell={w[0]}, s_lo={w[1]}, ell_int_sum={e}")

# Enumerate compositions: n_half=1 means we have a half of length n_half=1
# with sum S_half=40, so half = (40,). Then full c = (40, 40).
# But d = 2*n_half = 2, so palindromic c[0] = c[1] = 20? Let's see.
print(f"\nGenerating palindromic compositions of n_half={n_half} length, sum={S_half}:")
n_total = count_compositions(n_half, S_half)
print(f"n_total palindromic = {n_total}")

n_seen = 0
for half_batch in generate_compositions_batched(n_half, S_half, batch_size=200_000):
    print(f"half_batch shape: {half_batch.shape}")
    for h in half_batch:
        c_full = np.concatenate([h, h[::-1]])
        n_seen += 1
        if n_seen <= 30:
            print(f"  c_full = {c_full.tolist()}, sum = {c_full.sum()}")
print(f"total enumerated = {n_seen}")
