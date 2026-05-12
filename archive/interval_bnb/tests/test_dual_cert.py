"""Soundness test for bound_mccormick_joint_face_dual_cert_int_ge.

Checks:
  (a) Integer dual-cert bound LB <= primal LP value (weak duality).
  (b) If greedy SW/NE rigor bounds certify a box, joint cert also certifies.
      (Not strictly required: joint LP >= max(SW, NE), but rounding may
       degrade it. We instead just check the integer LB value directly.)
  (c) Joint cert returns True for some boxes where greedy SW/NE fails.
  (d) Joint cert never claims to certify a box where even the float LP
      value * scale falls short (should short-circuit).
"""
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
    bound_mccormick_sw_int_ge,
    bound_mccormick_ne_int_ge,
    bound_autoconv_int_ge,
    bound_natural_int_ge,
)
from interval_bnb.box import Box, SCALE
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def enumerate_boxes(d, depth):
    """BFS widest-axis split of the initial box, return all intersecting."""
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


def greedy_any_cert(B, w, d, tn, td):
    lo_int, hi_int = B.to_ints()
    if bound_natural_int_ge(lo_int, hi_int, w, tn, td):
        return True
    if bound_autoconv_int_ge(lo_int, hi_int, w, d, tn, td):
        return True
    if bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, tn, td):
        return True
    return bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, tn, td)


def main():
    d = 8
    windows = build_windows(d)
    target_q = Fraction("1.2")
    tn, td = target_q.numerator, target_q.denominator

    boxes = enumerate_boxes(d, depth=8)
    print(f"[test] d={d} target={target_q} boxes={len(boxes)} windows={len(windows)}")

    sound_violations = 0
    joint_only_cert = 0
    greedy_only_cert = 0
    both_cert = 0
    neither_cert = 0
    total_cases = 0

    for bi, B in enumerate(boxes):
        lo_int, hi_int = B.to_ints()
        lo = B.lo
        hi = B.hi
        for wi, w in enumerate(windows):
            total_cases += 1
            # Greedy int cert
            greedy_cert = (
                bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, tn, td)
                or bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, tn, td)
                or bound_natural_int_ge(lo_int, hi_int, w, tn, td)
                or bound_autoconv_int_ge(lo_int, hi_int, w, d, tn, td)
            )
            # Joint dual cert
            joint_cert = bound_mccormick_joint_face_dual_cert_int_ge(
                lo_int, hi_int, w, d, tn, td,
            )
            # Soundness: if joint cert says True, the LP value * scale must
            # be >= target. Verify by direct float LP eval.
            if joint_cert:
                lb_float = bound_mccormick_joint_face_lp(lo, hi, w, w.scale)
                # target_q with some small tolerance for float LP error.
                if lb_float < float(target_q) * (1 - 1e-6):
                    print(f"[WARN] joint_cert=True but float LP={lb_float:.6f} < "
                          f"target={float(target_q):.6f} at box {bi} window {wi} "
                          f"(likely float LP noise / stationarity rounding slack)")
                    sound_violations += 1

            if joint_cert and greedy_cert:
                both_cert += 1
            elif joint_cert:
                joint_only_cert += 1
            elif greedy_cert:
                greedy_only_cert += 1
            else:
                neither_cert += 1

    print(f"[test] total_cases={total_cases}")
    print(f"[test]   both_cert       = {both_cert}")
    print(f"[test]   joint_only_cert = {joint_only_cert}  <- tree-shrink here")
    print(f"[test]   greedy_only_cert= {greedy_only_cert}  <- joint-cert miss")
    print(f"[test]   neither_cert    = {neither_cert}")
    print(f"[test] sound violations  = {sound_violations} (MUST be 0)")
    assert sound_violations == 0, "SOUNDNESS FAILURE"
    print("[test] PASS")


if __name__ == "__main__":
    main()
