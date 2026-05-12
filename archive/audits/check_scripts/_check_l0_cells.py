"""Find the 25 L0 cells at (n_half=1, m=20, c=1.281).

L0 = first level of cascade — enumerate compositions (length d0=2, sum=80)
and apply F (variant F: LP-tight Δ_BB) to prune. Whatever survives F is
"L0 survivors" for the cascade.
"""
import sys, os
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

import numpy as np
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs

n_half = 1
m = 20
c_target = 1.281
d = 2 * n_half  # = 2
S = 4 * n_half * m  # = 80

# Enumerate ALL (non-palindromic) compositions of length d=2, sum=80
all_comps = []
for k in range(0, S + 1):
    all_comps.append([k, S - k])

batch = np.array(all_comps, dtype=np.int32)
print(f"Total compositions of length {d}, sum {S}: {len(all_comps)}")

# Apply F-prune
sF = prune_F(batch, n_half, m, c_target)
print(f"After F-prune: {int(np.sum(sF))} survivors out of {len(all_comps)}")

# Print survivors
windows, ell_int_sums = _build_windows(d)
sigmas = _enum_balanced_signs(d)
print(f"\nF-survivors:")
f_surv_list = []
for i, c in enumerate(batch):
    if sF[i]:
        # Also try Q
        q_pruned = prune_Q_one(c, windows, ell_int_sums, sigmas,
                               n_half, m, c_target, margin=1e-9)
        f_surv_list.append((i, c.tolist(), q_pruned))

print(f"Total F-survivors: {len(f_surv_list)}")
q_surv = [(i, c) for (i, c, q) in f_surv_list if not q]
print(f"Total Q-survivors among F-survivors: {len(q_surv)}")
for i, c, q in f_surv_list:
    marker = " (Q-pruned)" if q else " (Q-survivor)"
    print(f"  c={c}{marker}")
