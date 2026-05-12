"""Audit stuck_d10_master_queue.npz: how many of the 748 dumped boxes
would certify against target = 12/10 on each tier of the integer rigor
gate cascade?

Cascade order (cheapest -> most expensive):
    1. bound_natural_int_ge       (sum lo_i lo_j over W support)
    2. bound_autoconv_int_ge      (1 - complement-of-W on hi)
    3. bound_mccormick_sw_int_ge  (SW-face LP)
    4. bound_mccormick_ne_int_ge  (NE-face LP)

A box "certifies at tier T" iff some window passes the int_ge test at
tier T, and no earlier tier already certified.
"""
from __future__ import annotations

from fractions import Fraction
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from interval_bnb.windows import build_windows
from interval_bnb.box import Box
from interval_bnb.bound_eval import (
    bound_natural_int_ge,
    bound_autoconv_int_ge,
    bound_mccormick_sw_int_ge,
    bound_mccormick_ne_int_ge,
)


D = 10
TARGET = Fraction(12, 10)
TGT_NUM = TARGET.numerator
TGT_DEN = TARGET.denominator


def cert_tier(lo_int, hi_int, windows) -> int:
    """Return 1/2/3/4 for the tier that first certifies; 0 if none."""
    # Tier 1: natural
    for w in windows:
        if bound_natural_int_ge(lo_int, hi_int, w, TGT_NUM, TGT_DEN):
            return 1
    # Tier 2: autoconv
    for w in windows:
        if bound_autoconv_int_ge(lo_int, hi_int, w, D, TGT_NUM, TGT_DEN):
            return 2
    # Tier 3: McCormick SW
    for w in windows:
        if bound_mccormick_sw_int_ge(lo_int, hi_int, w, D, TGT_NUM, TGT_DEN):
            return 3
    # Tier 4: McCormick NE
    for w in windows:
        if bound_mccormick_ne_int_ge(lo_int, hi_int, w, D, TGT_NUM, TGT_DEN):
            return 4
    return 0


def main() -> None:
    data = np.load(os.path.join(_HERE, "stuck_d10_master_queue.npz"))
    lo_arr = data["lo"]
    hi_arr = data["hi"]
    depths = data["depths"]
    n = lo_arr.shape[0]
    assert lo_arr.shape == (n, D)

    print(f"Loaded {n} stuck boxes (d={D}, target={TARGET}).")
    windows = build_windows(d=D)
    print(f"Built {len(windows)} windows.")

    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    stuck_idx = []
    for k in range(n):
        B = Box(lo=lo_arr[k].astype(np.float64), hi=hi_arr[k].astype(np.float64))
        lo_int, hi_int = B.to_ints()
        t = cert_tier(lo_int, hi_int, windows)
        counts[t] += 1
        if t == 0:
            stuck_idx.append(k)
        if (k + 1) % 50 == 0:
            print(f"  ... {k+1}/{n} processed (stuck so far: {counts[0]})")

    print("\n=== Tier-by-tier certification counts ===")
    print(f"  tier 1 (natural)       : {counts[1]:4d} / {n}")
    print(f"  tier 2 (autoconv)      : {counts[2]:4d} / {n}")
    print(f"  tier 3 (McCormick SW)  : {counts[3]:4d} / {n}")
    print(f"  tier 4 (McCormick NE)  : {counts[4]:4d} / {n}")
    print(f"  STUCK (none cert)      : {counts[0]:4d} / {n}")
    cum = counts[1] + counts[2] + counts[3] + counts[4]
    print(f"  cumulative cert        : {cum:4d} / {n}  ({100.0*cum/n:.2f}%)")

    if stuck_idx:
        print(f"\nFirst {min(3, len(stuck_idx))} stuck-box indices: {stuck_idx[:3]}")
        for k in stuck_idx[:3]:
            B = Box(lo=lo_arr[k].astype(np.float64), hi=hi_arr[k].astype(np.float64))
            w = (B.hi - B.lo)
            print(f"  box {k}: depth={int(depths[k])}, "
                  f"max_w={float(w.max()):.3e}, min_w={float(w.min()):.3e}")
    else:
        print("\nAll dumped boxes certify on the cheap-tier cascade.")


if __name__ == "__main__":
    main()
