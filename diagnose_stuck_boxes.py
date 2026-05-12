"""Inspect where the 0.94% stuck volume is located at d=10 t=1.208.

Runs the BnB with verbose box-tracking, dumps the in_flight boxes at end
and shows their centroid + max_W TV_W.
"""
import os
import sys
import numpy as np
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.bnb import branch_and_bound
from interval_bnb.box import Box, SCALE
from interval_bnb.bound_eval import _adjacency_matrix
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def main():
    d = 10
    target = '1.208'
    target_q = Fraction(target)

    # We'll do a SERIAL bnb with stuck-box tracking
    print(f"Serial BnB d={d} target={target}, dumping unprovable boxes...")

    # Patch bnb to dump failing box on FAIL
    sym = half_simplex_cuts(d)
    initial = Box.initial(d, sym)

    # We'll just run BnB and inspect failure box
    res = branch_and_bound(
        d=d, target_c=target,
        max_nodes=10_000_000, time_budget_s=120,
        verbose=False,
    )
    print(f"\nResult: success={res.success}, nodes={res.stats.nodes_processed}")
    print(f"Max depth: {res.stats.max_depth}")
    if res.failing_box is not None:
        B = res.failing_box
        print(f"\nFailing box:")
        print(f"  lo: {B.lo}")
        print(f"  hi: {B.hi}")
        print(f"  width: {B.max_width():.4e}")
        center = (B.lo + B.hi) * 0.5
        center = center / center.sum()
        print(f"  center (norm): {center}")
        # Compute max_W TV_W at center and at corners of box
        windows = build_windows(d)
        tvs = []
        for w in windows:
            A = _adjacency_matrix(w, d)
            tv = w.scale * float(center @ A @ center)
            tvs.append(tv)
        tvs_sorted = sorted(enumerate(tvs), key=lambda x: -x[1])
        print(f"  Top-5 TV at center: ")
        for wi, tv in tvs_sorted[:5]:
            w = windows[wi]
            print(f"    W(ell={w.ell}, s={w.s_lo}): TV={tv:.6f}")
        print(f"  failing_lb={res.failing_lb:.6f}, target={float(target_q):.6f}, "
              f"gap={float(target_q) - res.failing_lb:.6f}")

if __name__ == "__main__":
    main()
