"""Quick LP-timing test for Q at d=14: build window data, run LP on a few comps."""
import os, sys, time
import numpy as np
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from _QN_bench import precompute_window_data, _qn_bound_lp
from _Q_bench import _q_bound_lp

n_half = 7
m = 5
c_target = 1.28
d = 2 * n_half
S_full = 4 * n_half * m  # = 140

print(f"=== d=14 LP timing ===")
print(f"n_half={n_half}, m={m}, d={d}, S_full={S_full}")

t0 = time.time()
windows, ell_int_sums, sigmas, m_W_arr, _ = precompute_window_data(d, n_half)
print(f"precompute: {time.time()-t0:.2f}s, n_win={len(windows)}, n_sigma={len(sigmas)}")

# A few sample compositions (palindromic, half-sum = 2nm = 70)
half_sum = 2 * n_half * m
samples = [
    # palindromic, sum = 4nm = 140
    np.array([10] * 14, dtype=np.int32),  # uniform, sum=140
    # Concentrated
    np.array([20, 5, 5, 5, 5, 5, 25, 25, 5, 5, 5, 5, 5, 20], dtype=np.int32),  # =140
    # Asymmetric-like (palindrome)
    np.array([1, 4, 7, 10, 13, 16, 19, 19, 16, 13, 10, 7, 4, 1], dtype=np.int32),  # =140
    # Very concentrated
    np.array([35, 0, 0, 0, 0, 0, 35, 35, 0, 0, 0, 0, 0, 35], dtype=np.int32),  # =140
]

print("\nSample LP runs:")
for c_int in samples:
    if c_int.sum() != S_full:
        print(f"  skip c_int.sum()={c_int.sum()} != {S_full}")
        continue
    t1 = time.time()
    tQ, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target)
    tQ_t = time.time() - t1
    t2 = time.time()
    tQN, _ = _qn_bound_lp(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target, m_W_arr)
    tQN_t = time.time() - t2
    print(f"  c={c_int.tolist()}: Q-excess={tQ:+.4f} [{1000*tQ_t:.1f}ms], "
          f"QN-excess={tQN:+.4f} [{1000*tQN_t:.1f}ms]")
