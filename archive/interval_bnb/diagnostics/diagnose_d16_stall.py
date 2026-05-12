"""Diagnostic: run short BnB at d=16 t=1.25, count how many rigor-retry
boxes would be closed by the joint-face dual-cert vs. stay unclosed.

If joint-cert closes a meaningful fraction of rigor-retry boxes at
moderate depth, then wiring joint-cert into rigor_replay fixes the
stall. If not, the stall has a different cause.
"""
from __future__ import annotations

import os
import sys
import time
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bound_eval import (
    batch_bounds_full,
    batch_bounds_rank1_hi,
    batch_bounds_rank1_lo,
    bound_autoconv_int_ge,
    bound_mccormick_joint_face_dual_cert_int_ge,
    bound_mccormick_ne_int_ge,
    bound_mccormick_sw_int_ge,
    bound_natural_int_ge,
    window_tensor,
)
from interval_bnb.box import Box
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def greedy_cert(lo_int, hi_int, w, d, tn, td):
    if bound_natural_int_ge(lo_int, hi_int, w, tn, td):
        return True
    if bound_autoconv_int_ge(lo_int, hi_int, w, d, tn, td):
        return True
    if bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, tn, td):
        return True
    return bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, tn, td)


def main():
    d = int(os.environ.get("DIAG_D", "16"))
    target_str = os.environ.get("DIAG_TARGET", "1.25")
    time_budget = float(os.environ.get("DIAG_TIME", "90"))
    min_depth = int(os.environ.get("DIAG_MIN_DEPTH", "18"))
    max_samples = int(os.environ.get("DIAG_MAX_SAMPLES", "200"))
    target_q = Fraction(target_str)
    target_f = float(target_q)
    tn, td = target_q.numerator, target_q.denominator

    sym = half_simplex_cuts(d)
    initial = Box.initial(d, sym)
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)

    # DFS. Collect boxes where:
    #   lb_fast >= target_f  AND  greedy_cert fails  AND  depth >= min_depth
    # For each, try joint_cert and count.
    stack = [(initial, 0, None, -1, None)]
    t0 = time.time()
    nodes = 0
    retry_boxes = 0      # failed greedy at depth >= min_depth
    joint_saves = 0      # joint-cert closed one that greedy missed
    joint_miss = 0       # joint-cert also failed
    total_retry = 0      # all greedy failures (any depth)
    samples_done = 0

    while stack:
        if time.time() - t0 > time_budget:
            print(f"[diag] time budget {time_budget}s exhausted")
            break
        if samples_done >= max_samples:
            print(f"[diag] sample cap {max_samples} reached")
            break
        B, depth, parent_cache, changed_k, which_end = stack.pop()
        nodes += 1
        if not B.intersects_simplex():
            continue
        if parent_cache is None:
            lb_fast, w_idx, which, _, my_cache = batch_bounds_full(
                B.lo, B.hi, A_tensor, scales, target_f,
            )
        elif which_end == "lo":
            lb_fast, w_idx, which, _, my_cache = batch_bounds_rank1_lo(
                A_tensor, scales, parent_cache, B.lo, changed_k, target_f,
            )
        else:
            lb_fast, w_idx, which, _, my_cache = batch_bounds_rank1_hi(
                A_tensor, scales, parent_cache, B.hi, changed_k, target_f,
            )
        if lb_fast >= target_f and w_idx >= 0:
            w = windows[w_idx]
            lo_int, hi_int = B.to_ints()
            if greedy_cert(lo_int, hi_int, w, d, tn, td):
                continue  # cert'd, done
            total_retry += 1
            if depth >= min_depth:
                retry_boxes += 1
                samples_done += 1
                t_lp0 = time.time()
                ok = bound_mccormick_joint_face_dual_cert_int_ge(
                    lo_int, hi_int, w, d, tn, td,
                )
                lp_ms = (time.time() - t_lp0) * 1000
                if ok:
                    joint_saves += 1
                    tag = "JOINT-SAVE"
                else:
                    joint_miss += 1
                    tag = "joint-miss"
                print(f"[diag] depth={depth} nodes={nodes} lb_fast={lb_fast:.6f} "
                      f"w={w_idx} {tag} lp_ms={lp_ms:.1f}")
                continue  # don't recurse further on sampled box
            # below min_depth: continue to split (keep searching for
            # deeper retry boxes)
        # split
        if B.max_width() < 1e-12:
            continue
        axis = B.widest_axis()
        left, right = B.split(axis)
        if right.intersects_simplex():
            stack.append((right, depth + 1, my_cache, axis, "lo"))
        if left.intersects_simplex():
            stack.append((left, depth + 1, my_cache, axis, "hi"))

    elapsed = time.time() - t0
    print()
    print(f"[diag] d={d} target={target_str} elapsed={elapsed:.1f}s")
    print(f"[diag] nodes_processed = {nodes}")
    print(f"[diag] total_retry (any depth) = {total_retry}")
    print(f"[diag] retry_boxes at depth>={min_depth} = {retry_boxes}")
    print(f"[diag]   joint_saves = {joint_saves}  <- joint-cert closes stall box")
    print(f"[diag]   joint_miss  = {joint_miss}   <- still stuck even with joint")
    if retry_boxes:
        save_rate = 100 * joint_saves / retry_boxes
        print(f"[diag]   save rate   = {save_rate:.1f}%")


if __name__ == "__main__":
    main()
