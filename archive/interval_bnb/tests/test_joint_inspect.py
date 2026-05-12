"""Inspect individual joint LP values vs greedy."""
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
    bound_mccormick_joint_face_lp,
    window_tensor,
    batch_bounds_full,
    _adjacency_matrix,
)
from interval_bnb.box import Box
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def _mccormick_greedy_sw(lo, hi, w):
    d = len(lo)
    A = _adjacency_matrix(w, d)
    Alo = A @ lo
    g = 2.0 * Alo
    c0 = -float(lo @ Alo)
    lo_sum = float(lo.sum())
    remaining = 1.0 - lo_sum
    val = float(g @ lo)
    if remaining > 0.0:
        order = np.argsort(g, kind="stable")
        for i in order:
            if remaining <= 0.0:
                break
            cap = hi[i] - lo[i]
            if cap <= remaining:
                val += g[i] * cap
                remaining -= cap
            else:
                val += g[i] * remaining
                remaining = 0.0
    return w.scale * (val + c0)


def _mccormick_greedy_ne(lo, hi, w):
    d = len(lo)
    A = _adjacency_matrix(w, d)
    Ahi = A @ hi
    g = 2.0 * Ahi
    c0 = -float(hi @ Ahi)
    lo_sum = float(lo.sum())
    remaining = 1.0 - lo_sum
    val = float(g @ lo)
    if remaining > 0.0:
        order = np.argsort(g, kind="stable")
        for i in order:
            if remaining <= 0.0:
                break
            cap = hi[i] - lo[i]
            if cap <= remaining:
                val += g[i] * cap
                remaining -= cap
            else:
                val += g[i] * remaining
                remaining = 0.0
    return w.scale * (val + c0)


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
    d = int(os.environ.get("D", "12"))
    depth = int(os.environ.get("DEPTH", "20"))
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)
    target_f = float(os.environ.get("TARGET", "1.2"))

    boxes = enumerate_boxes(d, depth=depth)
    print(f"[inspect] d={d} boxes={len(boxes)}")
    shown = 0
    for bi, B in enumerate(boxes):
        lb_fast, w_idx, which, _, _ = batch_bounds_full(
            B.lo, B.hi, A_tensor, scales, target_f,
        )
        if lb_fast >= target_f:
            continue  # would certify; skip
        w = windows[w_idx]
        sw = _mccormick_greedy_sw(B.lo, B.hi, w)
        ne = _mccormick_greedy_ne(B.lo, B.hi, w)
        joint = bound_mccormick_joint_face_lp(B.lo, B.hi, w, w.scale)
        print(f"[box {bi}] winning window ell={w.ell} s_lo={w.s_lo}  "
              f"lb_fast={lb_fast:.5f} ({which})  SW={sw:.5f}  NE={ne:.5f}  joint={joint:.5f}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
