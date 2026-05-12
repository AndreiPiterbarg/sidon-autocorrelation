"""Smoke test: just run F at n=6, m=10, c=1.28 to find F survivor count.

If F << many, then run Q+QN on the F-survivors.
"""
import os, sys, time
import numpy as np
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _N_bench import precompute_op_norm_restricted
from _FN_bench import prune_FN
from _Q_bench import prune_Q_one
from _QN_bench import precompute_window_data, prune_QN_one

n_half = 6
m = 10
c_target = 1.28
d = 2 * n_half  # 12
S_half = 2 * n_half * m  # 120

print(f"=== F-only count at d=12, m=10, c=1.28 ===")
print(f"Total palindromic comps: {count_compositions(n_half, S_half):,}")

t0 = time.time()
windows, ell_int_sums, sigmas, m_W_arr, _ = precompute_window_data(d, n_half)
conv_len = 2 * d - 1; max_ell = 2 * d
op_rest, n_pairs_arr = precompute_op_norm_restricted(d, max_ell, conv_len)
op_rest_d = op_rest * d
ell_int_arr = np.empty(conv_len, dtype=np.int64)
two_n = 2 * n_half
for k in range(conv_len):
    d_idx = abs((k + 1) - two_n)
    v = max(0, two_n - d_idx)
    ell_int_arr[k] = v
ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
for k in range(conv_len):
    ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]
print(f"Precompute: {time.time()-t0:.2f}s, n_win={len(windows)}, n_sigma={len(sigmas)}")

# Warm
warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = 2 * m
prune_F(warm, n_half, m, c_target)
prune_FN(warm, n_half, m, c_target, ell_prefix, op_rest_d)

n_proc = 0
surv_F = 0
surv_FN = 0
F_survivors = []  # store actual survivor comps
t_F = t_FN = 0.0
t_run = time.time()

for half_batch in generate_compositions_batched(n_half, S_half, batch_size=500_000):
    batch = np.empty((len(half_batch), d), dtype=np.int32)
    batch[:, :n_half] = half_batch
    batch[:, n_half:] = half_batch[:, ::-1]
    n_proc += len(batch)

    ta = time.time()
    sF = prune_F(batch, n_half, m, c_target)
    t_F += time.time() - ta
    surv_F += int(sF.sum())

    tb = time.time()
    sFN = prune_FN(batch, n_half, m, c_target, ell_prefix, op_rest_d)
    t_FN += time.time() - tb
    surv_FN += int(sFN.sum())

    # collect F survivors
    if sF.any():
        F_survivors.extend(batch[sF].tolist())

    if n_proc % 10_000_000 < 500_000:
        print(f"  processed {n_proc:,} ({100.0*n_proc/234531275:.1f}%), "
              f"F-surv={surv_F}, FN-surv={surv_FN}, "
              f"t_F={t_F:.1f}s, t_FN={t_FN:.1f}s")

t_run = time.time() - t_run
print(f"\nTotal: F={surv_F}, FN={surv_FN}, FN-extra over F={surv_F-surv_FN}")
print(f"Wall: {t_run:.1f}s,   t_F={t_F:.1f}s, t_FN={t_FN:.1f}s")

# Run Q + QN on F-survivors
if F_survivors:
    print(f"\n=== Running Q+QN on {len(F_survivors)} F-survivors ===")
    t1 = time.time()
    Q_extra = QN_extra = 0
    for c_int in F_survivors:
        c_arr = np.array(c_int, dtype=np.int32)
        if prune_Q_one(c_arr, windows, ell_int_sums, sigmas, n_half, m, c_target):
            Q_extra += 1
            QN_extra += 1  # if Q prunes, so does QN (QN <= Q)
        elif prune_QN_one(c_arr, windows, ell_int_sums, sigmas, n_half, m, c_target, m_W_arr):
            QN_extra += 1
    t_lp = time.time() - t1
    print(f"  Q-extra over F: {Q_extra}    Q surv: {len(F_survivors) - Q_extra}")
    print(f"  QN-extra over F: {QN_extra}   QN surv: {len(F_survivors) - QN_extra}")
    print(f"  t_LP: {t_lp:.1f}s, ms/LP: {1000*t_lp/len(F_survivors):.1f}")
