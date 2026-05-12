"""Measure the gap between joint-LP and max(SW, NE) on real BnB-like boxes."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from fractions import Fraction

import numpy as np

from interval_bnb.bound_eval import (
    bound_mccormick_joint_face_dual_cert_int_ge,
    bound_mccormick_joint_face_lp,
    window_tensor,
    batch_bounds_full,
)
from interval_bnb.box import Box
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def enumerate_boxes(d, depth):
    sym = half_simplex_cuts(d)
    initial = Box.initial(d, sym)
    boxes = [(initial, 0)]
    out = []
    while boxes:
        B, cur_depth = boxes.pop()
        if not B.intersects_simplex():
            continue
        if cur_depth >= depth:
            out.append(B)
            continue
        ax = B.widest_axis()
        left, right = B.split(ax)
        boxes.append((left, cur_depth + 1))
        boxes.append((right, cur_depth + 1))
    return out


def main():
    d = 8
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)
    target_q = Fraction("1.2")
    target_f = float(target_q)
    tn, td = target_q.numerator, target_q.denominator

    for depth in (4, 6, 8, 10, 12):
        boxes = enumerate_boxes(d, depth=depth)
        gap_values = []
        cert_via_joint_only = 0
        total_would_split = 0
        total_boxes = 0
        for B in boxes:
            total_boxes += 1
            # Compute best greedy lb_fast
            lb_fast, w_idx, which, _, _ = batch_bounds_full(
                B.lo, B.hi, A_tensor, scales, target_f,
            )
            if lb_fast >= target_f:
                continue
            total_would_split += 1
            # Pick the top window and try joint LP
            w = windows[w_idx]
            joint_f = bound_mccormick_joint_face_lp(B.lo, B.hi, w, w.scale)
            gap = joint_f - lb_fast
            gap_values.append(gap)
            if joint_f >= target_f:
                # Would the int dual cert also certify?
                lo_int, hi_int = B.to_ints()
                if bound_mccormick_joint_face_dual_cert_int_ge(
                    lo_int, hi_int, w, d, tn, td,
                ):
                    cert_via_joint_only += 1

        if gap_values:
            gap_arr = np.array(gap_values)
            print(f"[depth={depth}] boxes={total_boxes}, would_split={total_would_split}, "
                  f"cert_via_joint_only={cert_via_joint_only}  "
                  f"gap_mean={gap_arr.mean():.5f} gap_median={np.median(gap_arr):.5f} "
                  f"gap_max={gap_arr.max():.5f}")
        else:
            print(f"[depth={depth}] boxes={total_boxes}, would_split={total_would_split} (all cert greedy)")


if __name__ == "__main__":
    main()
