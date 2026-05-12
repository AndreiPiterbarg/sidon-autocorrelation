"""Diagnostic: at d=16 t=1.25, descend DFS until depth>=min_depth and
sample boxes where lb_fast < target_f. For each, compute joint-face
LP (tighter float bound). Report: does joint-face LP cross target on
these "stuck in float" boxes?

If YES: float bound is the blocker; need joint-face in float path.
If NO: bound is fundamentally loose at d=16; tree must split deeper.
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
    bound_mccormick_joint_face_lp,
    window_tensor,
)
from interval_bnb.box import Box
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


def main():
    d = int(os.environ.get("DIAG_D", "16"))
    target_str = os.environ.get("DIAG_TARGET", "1.25")
    time_budget = float(os.environ.get("DIAG_TIME", "120"))
    min_depth = int(os.environ.get("DIAG_MIN_DEPTH", "25"))
    max_samples = int(os.environ.get("DIAG_MAX_SAMPLES", "50"))
    target_q = Fraction(target_str)
    target_f = float(target_q)

    sym = half_simplex_cuts(d)
    initial = Box.initial(d, sym)
    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)

    # DFS. When we reach depth >= min_depth AND lb_fast < target_f,
    # SAMPLE the box: compute joint LP for TOP window and for top-3
    # windows, compare to target.
    stack = [(initial, 0, None, -1, None)]
    t0 = time.time()
    nodes = 0
    samples = []
    deep_reached = 0

    while stack:
        if time.time() - t0 > time_budget:
            break
        if len(samples) >= max_samples:
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

        if depth >= min_depth:
            deep_reached += 1
            if lb_fast < target_f:
                # Sample this box: compute joint-LP on TOP window only
                # (keep it fast).
                w = windows[w_idx]
                t_lp = time.time()
                joint_f = bound_mccormick_joint_face_lp(B.lo, B.hi, w, w.scale)
                lp_ms = (time.time() - t_lp) * 1000
                samples.append({
                    "depth": depth,
                    "lb_fast": lb_fast,
                    "which": which,
                    "w_idx": w_idx,
                    "joint_lp": joint_f,
                    "joint_gap": joint_f - lb_fast,
                    "joint_crosses_target": joint_f >= target_f,
                    "max_width": B.max_width(),
                    "lp_ms": lp_ms,
                })
                # Don't recurse into sampled boxes.
                continue

        if lb_fast >= target_f:
            continue  # cert'd (skip; float path says OK)

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
    print(f"[diag] nodes_processed={nodes} deep_reached(>={min_depth})={deep_reached}")
    print(f"[diag] samples(lb_fast<target at deep depth) = {len(samples)}")
    if samples:
        joint_saves = sum(1 for s in samples if s["joint_crosses_target"])
        gaps = sorted(s["joint_gap"] for s in samples)
        lbs = sorted(s["lb_fast"] for s in samples)
        depths = sorted(s["depth"] for s in samples)
        lp_mss = sorted(s["lp_ms"] for s in samples)
        print(f"[diag] joint_crosses_target = {joint_saves}/{len(samples)}")
        print(f"[diag] lb_fast min={lbs[0]:.6f} median={lbs[len(lbs)//2]:.6f} "
              f"max={lbs[-1]:.6f} (target={target_f})")
        print(f"[diag] joint_gap min={gaps[0]:.6f} median={gaps[len(gaps)//2]:.6f} "
              f"max={gaps[-1]:.6f}")
        print(f"[diag] depth min={depths[0]} median={depths[len(depths)//2]} "
              f"max={depths[-1]}")
        print(f"[diag] lp_ms min={lp_mss[0]:.1f} median={lp_mss[len(lp_mss)//2]:.1f} "
              f"max={lp_mss[-1]:.1f}")
        # show a few
        print()
        print("[diag] first 10 samples (depth, lb_fast, joint_lp, joint_gap, max_w):")
        for s in samples[:10]:
            print(f"  d={s['depth']:3d} lb_fast={s['lb_fast']:.6f} "
                  f"joint={s['joint_lp']:.6f} gap={s['joint_gap']:+.6f} "
                  f"maxw={s['max_width']:.2e} lp={s['lp_ms']:.1f}ms")


if __name__ == "__main__":
    main()
