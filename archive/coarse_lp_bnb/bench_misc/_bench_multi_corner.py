"""Benchmark multi-corner anchor cert rate vs smart-W centroid alone
on 30 simulated d=22 stuck-like boxes at target=1.281.

Box geometry:
  16 boundary axes: lo=0, hi=0.05
  6 free axes:      lo=0.05, hi=0.25  (with random center)

Reports:
  - cert rate: multi-corner vs smart-W centroid alone
  - per-box cost (mean ms, multi-corner)
"""
from __future__ import annotations

import os
import sys
import time
from fractions import Fraction
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from interval_bnb.bound_anchor import (
    build_centroid_anchor_cache, bound_anchor_centroid_int_ge,
    bound_anchor_multi_corner_int_ge,
)
from interval_bnb.box import SCALE as _SCALE
from interval_bnb.windows import build_windows


def main():
    d = 22
    windows = build_windows(d)
    print(f"[bench] d={d} |W|={len(windows)} — building cache...")
    t0 = time.time()
    cache = build_centroid_anchor_cache(d, windows=windows)
    print(f"[bench] cache built in {time.time() - t0:.2f}s")

    target_q = Fraction(1281, 1000)

    rng = np.random.default_rng(20260901)
    n_total = 0
    n_centroid = 0
    n_multi = 0
    centroid_times: list[float] = []
    multi_times: list[float] = []

    for trial in range(30):
        lo_f = np.zeros(d)
        hi_f = np.zeros(d)
        for i in range(16):
            lo_f[i] = 0.0
            hi_f[i] = 0.05
        for i in range(16, 22):
            mid = rng.uniform(0.07, 0.20)
            lo_f[i] = max(0.05, mid - 0.10)
            hi_f[i] = min(0.30, mid + 0.05)
            if hi_f[i] - lo_f[i] < 0.05:
                hi_f[i] = lo_f[i] + 0.05
        if lo_f.sum() > 1.0 or hi_f.sum() < 1.0:
            print(f"[trial {trial}] infeasible (lo_sum={lo_f.sum():.3f}, hi_sum={hi_f.sum():.3f}); skipped")
            continue
        lo_int = [int(round(x * _SCALE)) for x in lo_f]
        hi_int = [int(round(x * _SCALE)) for x in hi_f]
        if sum(lo_int) > _SCALE or sum(hi_int) < _SCALE:
            continue
        n_total += 1

        t1 = time.time()
        cc = bound_anchor_centroid_int_ge(
            lo_int, hi_int,
            target_q.numerator, target_q.denominator, cache,
        )
        centroid_times.append(time.time() - t1)
        if cc:
            n_centroid += 1

        t2 = time.time()
        cm = bound_anchor_multi_corner_int_ge(
            lo_int, hi_int,
            target_q.numerator, target_q.denominator, cache,
        )
        multi_times.append(time.time() - t2)
        if cm:
            n_multi += 1

        print(f"[trial {trial}] centroid={cc} multi={cm} "
              f"(cent={centroid_times[-1]*1000:.1f}ms multi={multi_times[-1]*1000:.1f}ms)")

    print(f"\n=== Results ===")
    print(f"N feasible boxes: {n_total}/30")
    print(f"smart-W centroid certs: {n_centroid}/{n_total} = {n_centroid/max(n_total,1):.1%}")
    print(f"multi-corner certs:     {n_multi}/{n_total} = {n_multi/max(n_total,1):.1%}")
    print(f"")
    print(f"Per-box mean cost (ms):")
    print(f"  smart-W centroid: {1000 * np.mean(centroid_times):.1f}")
    print(f"  multi-corner:     {1000 * np.mean(multi_times):.1f}")


if __name__ == "__main__":
    main()
