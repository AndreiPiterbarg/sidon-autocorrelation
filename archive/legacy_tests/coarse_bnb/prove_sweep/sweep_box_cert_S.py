"""Sweep S values for c_target=1.29 to find the minimum S where
box certification passes at d=2 (L0 dimension).

Runs box cert on ALL grid cells (exhaustive, not sampled), regardless
of whether the cascade pruned them.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from coarse_cascade_prover import (
    _box_certify_cell, compute_xcap,
)

C_TARGET = 1.29
D = 2  # L0 dimension


def exhaustive_box_cert(S, c_target, d):
    """Test ALL valid compositions of S into d parts for box cert."""
    delta = 1.0 / S
    xcap = compute_xcap(c_target, S, d)
    n_cert = 0
    n_fail = 0
    worst_tv = 1e30
    worst_comp = None

    # d=2: enumerate all (c0, c1) with c0+c1=S, 0<=ci<=xcap
    lo0 = max(0, S - xcap)
    hi0 = min(S, xcap)
    for c0 in range(lo0, hi0 + 1):
        c1 = S - c0
        if c1 < 0 or c1 > xcap:
            continue
        mu_center = np.array([c0 / S, c1 / S], dtype=np.float64)
        certified, min_tv = _box_certify_cell(mu_center, d, delta, c_target)
        if min_tv < worst_tv:
            worst_tv = min_tv
            worst_comp = (c0, c1)
        if certified:
            n_cert += 1
        else:
            n_fail += 1

    total = n_cert + n_fail
    return n_cert, n_fail, total, worst_tv, worst_comp


# Sweep S values
for S in range(10, 10001, 5):
    n_cert, n_fail, total, worst_tv, worst_comp = exhaustive_box_cert(S, C_TARGET, D)
    pct = n_cert / max(total, 1) * 100
    status = "PASS" if n_fail == 0 else "FAIL"
    print(f"S={S:5d}: {n_cert}/{total} certified ({pct:5.1f}%), "
          f"worst_tv={worst_tv:.6f}, worst_comp={worst_comp}, [{status}]",
          flush=True)

    if n_fail == 0:
        print(f"\n*** SUCCESS: S={S} passes exhaustive box cert at c_target={C_TARGET}, d={D} ***")
        break
else:
    print(f"\nNo S in [10, 10000] achieved 100% box cert for c_target={C_TARGET}, d={D}")
