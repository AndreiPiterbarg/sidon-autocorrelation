"""For each box that greedy fails to certify, try joint LP on ALL windows
to see if any window's joint LP can certify."""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from interval_bnb.bound_eval import (
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
    d = int(os.environ.get("D", "10"))
    depth = int(os.environ.get("DEPTH", "16"))
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)
    target_f = float(os.environ.get("TARGET", "1.2"))

    boxes = enumerate_boxes(d, depth=depth)
    failing = []
    for B in boxes:
        lb_fast, _, _, _, _ = batch_bounds_full(B.lo, B.hi, A_tensor, scales, target_f)
        if lb_fast < target_f:
            failing.append((B, lb_fast))
    print(f"[sweep] d={d} depth={depth} target={target_f} "
          f"total_boxes={len(boxes)} failing={len(failing)}")

    # For each failing box, try joint LP on all windows; report max joint value.
    max_gain = 0.0
    gains = []
    joint_cert_count = 0
    for bi, (B, lb_fast) in enumerate(failing[:50]):
        best_joint = float("-inf")
        best_joint_w = -1
        for wi, w in enumerate(windows):
            j = bound_mccormick_joint_face_lp(B.lo, B.hi, w, w.scale)
            if j > best_joint:
                best_joint = j
                best_joint_w = wi
        gain = best_joint - lb_fast
        gains.append(gain)
        if best_joint >= target_f:
            joint_cert_count += 1
    if gains:
        garr = np.array(gains)
        print(f"[sweep] on first {len(gains)} failing: "
              f"gain_mean={garr.mean():.5f}  gain_max={garr.max():.5f}  "
              f"pct_joint_cert={100*joint_cert_count/len(gains):.1f}%")


if __name__ == "__main__":
    main()
